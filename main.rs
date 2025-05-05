use clap::{Parser, Subcommand};
use glob::glob;
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::{f64, fmt};

// Open 5 letter words dictionary
fn open_dictionary<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let words: Vec<String> = reader.lines().filter_map(Result::ok).collect();

    Ok(words)
}

#[derive(PartialEq, Clone, Copy)]
enum MatchKind {
    NoMatch,
    Partial,
    Match,
}

impl fmt::Display for MatchKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MatchKind::NoMatch => "NoMatch",
            MatchKind::Partial => "Partial",
            MatchKind::Match => "Match",
        };
        write!(f, "{}", s)
    }
}

// Declare a custom match result
type MatchResult = [MatchKind; 5];

#[derive(Clone, Copy, Debug, PartialEq)]
struct WordEncoding {
    positions: [char; 5],  // Encode symbol position
    frequencies: [u8; 26], // Encode symbol frequency
}

impl WordEncoding {
    /// helper: A→0, B→1, … Z→25
    #[inline]
    fn idx(c: char) -> usize {
        (c.to_ascii_uppercase() as u8 - b'A') as usize
    }

    pub fn to_string(&self) -> String {
        self.positions.iter().collect()
    }

    pub fn from_string(word: &str) -> WordEncoding {
        let mut positions = ['A'; 5];
        let mut frequencies = [0u8; 26];

        for (i, c) in word.chars().enumerate() {
            let cu = c.to_ascii_uppercase();
            positions[i] = cu;
            frequencies[Self::idx(cu)] += 1;
        }

        WordEncoding {
            positions,
            frequencies,
        }
    }

    pub fn match_result(&self, other: &WordEncoding) -> MatchResult {
        let mut result = [MatchKind::NoMatch; 5];
        let mut remaining = other.frequencies; // local mutable copy

        for i in 0..5 {
            if self.positions[i] == other.positions[i] {
                result[i] = MatchKind::Match;
                remaining[Self::idx(self.positions[i])] -= 1;
            }
        }

        for i in 0..5 {
            if result[i] == MatchKind::NoMatch {
                let idx = Self::idx(self.positions[i]);
                if remaining[idx] > 0 {
                    result[i] = MatchKind::Partial;
                    remaining[idx] -= 1;
                }
            }
        }

        result
    }
}

#[derive(PartialEq)]
enum Policy {
    MaximizeEntropy,
    MinimizeScore,
}

struct WordleSolver {
    dictionary: Vec<WordEncoding>, // Dictionary as tuple of WordEncoding, sorted by rank. E.G. dictionary[0] is the word with the highest frequency
    policy: Policy,                // The policy of the algorithm
    expected_moves_curve: Vec<Bucket>, // The expected moves given an entropy (from our training)
    previous_guesses: Vec<WordEncoding>, // Track previous guesses

    // These are our state variables - should be updated on every iteration or guess
    prior: Vec<f64>, // P_W(w): The probability mass function of how plausible our word is the answer
    current_possibilities: Vec<usize>, // Set of current possibilities (W), stored as indices of elements in dictionary.

    // These are values derived from our state
    current_guess: Option<WordEncoding>,
    current_guess_entropy: f64,
    current_guess_match_result: Option<Vec<(MatchResult, f64)>>,
    current_guess_match_pattern_pd: Option<[f64; 243]>,
    current_expected_score: f64,
}

impl WordleSolver {
    pub fn intialise(
        dictionary_path: &String,
        policy: Policy,
        expected_moves_curve: Vec<Bucket>,
    ) -> Result<WordleSolver, String> {
        let mut solver: WordleSolver;

        let dictionary_result = open_dictionary(dictionary_path);
        if dictionary_result.is_err() {
            return Err(format!(
                "Error opening dictionary: {}",
                dictionary_result.unwrap_err()
            )
            .to_string());
        }

        let dictionary = dictionary_result.unwrap();
        let dictionary_len = dictionary.len();

        println!("Loaded dictionary with {} words", dictionary_len);

        solver = WordleSolver {
            dictionary: WordleSolver::compute_word_encodings(&dictionary),
            policy,
            previous_guesses: Vec::new(),
            prior: vec![0.0; dictionary_len],
            current_possibilities: (0..dictionary_len).collect(),
            current_guess: None,
            current_guess_entropy: 0.0,
            current_guess_match_result: None,
            current_guess_match_pattern_pd: None,
            current_expected_score: f64::INFINITY,
            expected_moves_curve: expected_moves_curve,
        };

        //  Update the prior in the solver before returning it
        solver.update_prior();

        Ok(solver)
    }

    pub fn reset(&mut self) {
        // Reset values
        self.current_guess = None;
        self.current_guess_entropy = 0.0;
        self.current_guess_match_result = None;
        self.current_guess_match_pattern_pd = None;
        self.current_expected_score = f64::INFINITY;
        self.previous_guesses.clear();

        // Reset possibilties
        self.current_possibilities = (0..self.dictionary.len()).collect();

        // Reset prior
        self.update_prior();
    }

    // Update our prior with the current possibilities
    pub fn update_prior(&mut self) {
        let mut weights = vec![0.0; self.dictionary.len()];
        let parametric_sigmoid = |x: f64, midpoint: f64, steepness: f64| -> f64 {
            1.0 / (1.0 + (steepness * (x - midpoint)).exp())
        };

        let mut sum_weight: f64 = 0.0;
        for w in self.current_possibilities.iter() {
            weights[*w] = parametric_sigmoid(*w as f64, 1500.0, 0.05);
            sum_weight += weights[*w]
        }

        // Update the prior probabilities
        for i in 0..self.prior.len() {
            self.prior[i] = weights[i] / sum_weight;
        }
    }

    pub fn guess<CheckFunction>(&mut self, callback: CheckFunction)
    where
        CheckFunction: Fn(&WordEncoding) -> MatchResult,
    {
        if let Some(some_guess) = &self.current_guess {
            self.previous_guesses.push(*some_guess);

            let actual_match = callback(some_guess);

            let keep_indices: Vec<usize> = self
                .current_guess_match_result
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .filter(|(_, val)| (**val).0 == actual_match)
                .map(|(index, _)| index)
                .collect();

            self.current_possibilities = keep_indices
                .iter()
                .map(|i| self.current_possibilities[*i])
                .collect();

            self.update_prior();
        }
    }

    pub fn step(&mut self) {
        self.current_guess = None;
        self.current_guess_entropy = 0.0;
        self.current_guess_match_result = None;
        self.current_guess_match_pattern_pd = None;
        self.current_expected_score = f64::INFINITY;

        // Calculate entropy of every possibilities
        for (i, guess) in self.dictionary.iter().enumerate() {
            // Do not repeat our guess
            if self.previous_guesses.contains(guess) {
                continue;
            }

            let mut match_results: Vec<(MatchResult, f64)> = Vec::new();

            for j in self.current_possibilities.iter() {
                let match_pattern = guess.match_result(&self.dictionary[*j]);
                match_results.push((match_pattern, self.prior[*j]))
            }

            let match_pattern_pd = WordleSolver::compute_match_pattern_pd(&match_results);
            let entropy = WordleSolver::compute_entropy(match_pattern_pd);

            if self.policy == Policy::MaximizeEntropy {
                if entropy > self.current_guess_entropy {
                    self.current_guess = Some(*guess);
                    self.current_guess_entropy = entropy;
                    self.current_guess_match_result = Some(match_results);
                    self.current_guess_match_pattern_pd = Some(match_pattern_pd);
                }
            } else if self.policy == Policy::MinimizeScore {
                // We really need to punish when the prior is zero - we only want to explore when prior is zero

                let expected_score = 1.0
                    + (1.0 - self.prior[i])
                        * self.compute_expected_score(
                            (self.current_possibilities.len() as f64).log2() - entropy,
                        );

                if expected_score < self.current_expected_score {
                    self.current_guess = Some(*guess);
                    self.current_guess_entropy = entropy;
                    self.current_guess_match_result = Some(match_results);
                    self.current_guess_match_pattern_pd = Some(match_pattern_pd);
                    self.current_expected_score = expected_score;
                }
            }
        }
    }

    fn compute_word_encodings(words: &Vec<String>) -> Vec<WordEncoding> {
        let mut encodings: Vec<WordEncoding> = Vec::new();

        // Compute encoding for each word
        for word in words {
            let encoding = WordEncoding::from_string(word);
            encodings.push(encoding);
        }

        encodings
    }

    // Compute the 'match pattern' probability distribution (pd), of a given word over the possibility
    fn compute_match_pattern_pd(match_results: &Vec<(MatchResult, f64)>) -> [f64; 243] {
        let mut sum: f64 = 0.0;
        let mut match_pattern_pd: [f64; 243] = [0.0; 243];

        for (match_result, likelihood) in match_results {
            // Compute the index
            let mut index: usize = 0;

            for i in 0..5 {
                match match_result[i] {
                    MatchKind::NoMatch => index += 0 * (3 as usize).pow(i as u32),
                    MatchKind::Partial => index += 1 * (3 as usize).pow(i as u32),
                    MatchKind::Match => index += 2 * (3 as usize).pow(i as u32),
                }
            }

            match_pattern_pd[index] += likelihood;
            sum += likelihood;
        }

        // Normalise
        for x in match_pattern_pd.iter_mut() {
            *x /= sum;
        }

        match_pattern_pd
    }

    fn compute_entropy<const N: usize>(pd: [f64; N]) -> f64 {
        let mut entropy: f64 = 0.0;
        for probabilty in pd.iter() {
            if *probabilty > 0.0 {
                entropy += -1.0 * (*probabilty) * (*probabilty).log2();
            }
        }

        entropy
    }

    fn compute_expected_score(&self, entropy: f64) -> f64 {
        if self.expected_moves_curve.is_empty() {
            // fallback: rough proxy = entropy itself
            entropy
        } else {
            interp_expected_moves(&self.expected_moves_curve, entropy)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Bucket {
    centre: f64,    // bucket midpoint (x axis)
    avg_moves: f64, // average moves‑remaining in this bucket
}

fn build_moves_histogram(glob_pattern: &str, bucket_width: f64) -> io::Result<Vec<Bucket>> {
    let mut sum_moves: HashMap<i64, f64> = HashMap::new();
    let mut counts: HashMap<i64, usize> = HashMap::new();

    for entry in glob(glob_pattern).map_err(|e| io::Error::new(io::ErrorKind::Other, e))? {
        let path = entry.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let file = std::fs::File::open(&path)?; // ordinary io::Error
        for line in io::BufReader::new(file).lines().skip(1) {
            let l = line?;
            let mut it = l.split(',');
            it.next(); // secret_idx
            let entropy: f64 = it.next().unwrap().parse().unwrap();
            let moves: f64 = it.next().unwrap().parse().unwrap();

            let idx = (entropy / bucket_width).floor() as i64;
            *sum_moves.entry(idx).or_insert(0.0) += moves;
            *counts.entry(idx).or_insert(0) += 1;
        }
    }

    let mut buckets: Vec<Bucket> = sum_moves
        .into_iter()
        .map(|(idx, sum)| {
            let count = counts[&idx];
            Bucket {
                centre: (idx as f64 + 0.5) * bucket_width,
                avg_moves: sum / count as f64,
            }
        })
        .collect();

    buckets.sort_by(|a, b| a.centre.partial_cmp(&b.centre).unwrap());
    Ok(buckets)
}

/// Linear interpolation (flat extrapolation) on the buckets.
fn interp_expected_moves(buckets: &[Bucket], entropy: f64) -> f64 {
    match buckets {
        [] => f64::NAN,
        [only] => only.avg_moves,
        _ => {
            if entropy <= buckets[0].centre {
                return buckets[0].avg_moves;
            }
            if entropy >= buckets.last().unwrap().centre {
                return buckets.last().unwrap().avg_moves;
            }
            for w in buckets.windows(2) {
                let (l, r) = (w[0], w[1]);
                if entropy >= l.centre && entropy <= r.centre {
                    let t = (entropy - l.centre) / (r.centre - l.centre);
                    return l.avg_moves + t * (r.avg_moves - l.avg_moves);
                }
            }
            unreachable!()
        }
    }
}

fn spawn_workers(requested: usize, kind: RunKind) {
    let logical = num_cpus::get();
    let n = if requested == 0 {
        logical
    } else {
        requested.min(logical)
    };

    println!("Spawning {n} {:?} workers…", kind);

    let mut children = Vec::new();
    for id in 0..n {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.arg(match kind {
            RunKind::Train => "train-worker",
            RunKind::Test => "test-worker",
        })
        .arg(id.to_string())
        .arg(n.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
        children.push(cmd.spawn().expect("spawn failed"));
    }
    for mut c in children {
        c.wait().unwrap();
    }
}

fn interactive_play() {
    let shards_glob = "./train/training_data*.csv";
    let mut curve = Vec::new();

    let policy: Policy;

    if glob::glob(shards_glob)
        .expect("bad glob pattern")
        .any(|res| res.as_ref().map(|p| p.is_file()).unwrap_or(false))
    {
        // at least one shard exists → build histogram & switch policy
        match build_moves_histogram(shards_glob, 0.20) {
            Ok(buckets) if !buckets.is_empty() => {
                curve = buckets;
                policy = Policy::MinimizeScore; // use data‑driven scoring
                println!("Loaded expected‑moves curve from training data ✅");
            }
            _ => {
                eprintln!(
                    "⚠️  Training data present but histogram build failed – using entropy policy"
                );
                policy = Policy::MaximizeEntropy;
            }
        }
    } else {
        policy = Policy::MaximizeEntropy; // no training data yet
    }

    // ------------------------------------------------------------ //
    // 2.  Create solver with chosen policy & curve                 //
    // ------------------------------------------------------------ //
    let mut solver = match WordleSolver::intialise(
        &"./words_5_letters.txt".to_string(),
        policy,
        curve, // <‑‑ pass curve (may be empty)
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to initialize WordleSolver: {e}");
            std::process::exit(1);
        }
    };

    while solver.current_possibilities.len() > 1 {
        let initial_possibilities = solver.current_possibilities.len();

        // Prime the initial guess using step()
        solver.step();

        if solver.current_guess.is_none() {
            eprintln!("failed to find solution: cannot generate next guess");
            std::process::exit(1);
        }

        let guess = solver.current_guess.as_ref().unwrap();

        println!(
            "Guess: {}, Expected #guesses: {}, Expected ΔEntropy: {}, Remaining Possibilities: {}",
            guess.to_string(),
            solver.current_expected_score,
            solver.current_guess_entropy,
            initial_possibilities
        );

        // Ask the user for feedback
        print!("Enter feedback (M = Match, P = Partial, N = No match, e.g. MPNPN): ");
        io::stdout().flush().unwrap();
        let mut feedback = String::new();
        io::stdin()
            .read_line(&mut feedback)
            .expect("Failed to read input");
        let feedback = feedback.trim().to_uppercase();

        if feedback.len() != 5 {
            eprintln!(
                "Feedback must be exactly 5 characters (M/P/N). Got: {}",
                feedback
            );
            std::process::exit(1);
        }

        // Parse feedback into MatchResult
        let parsed_feedback = match parse_feedback(&feedback) {
            Ok(parsed) => parsed,
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        };

        // Now filter possibilities manually using the parsed feedback
        solver.guess(|_| parsed_feedback);

        let actual_entropy: f64 = f64::log2(initial_possibilities as f64)
            - f64::log2(solver.current_possibilities.len() as f64);

        println!(
            "New Remaining Possibilities: {}, Actual ΔEntropy: {}",
            solver.current_possibilities.len(),
            actual_entropy
        );
    }

    println!(
        "Solution Found: {}",
        solver.dictionary[solver.current_possibilities[0]].to_string()
    );
}

/// Parse the user feedback string like "MPNPN" into MatchResult
fn parse_feedback(feedback: &str) -> Result<MatchResult, String> {
    let mut result = [MatchKind::NoMatch; 5];
    for (i, c) in feedback.chars().enumerate() {
        result[i] = match c {
            'M' => MatchKind::Match,
            'P' => MatchKind::Partial,
            'N' => MatchKind::NoMatch,
            _ => {
                return Err(format!(
                    "Invalid feedback character '{}'. Use only M, P, N.",
                    c
                ));
            }
        }
    }
    Ok(result)
}

/// Where should the worker write its shard?
#[derive(Debug, Clone, Copy)]
enum RunKind {
    Train, //  → ./train/training_data.{id}.csv
    Test,  //  → ./test/testing_data.{id}.csv
}

impl RunKind {
    fn dir(&self) -> &'static str {
        match self {
            RunKind::Train => "./train",
            RunKind::Test => "./test",
        }
    }
    fn shard_name(&self, id: usize) -> String {
        format!(
            "{}/{}_data.{}.csv",
            self.dir(),
            match self {
                RunKind::Train => "training",
                RunKind::Test => "testing",
            },
            id
        )
    }
}

fn run_generic_worker(kind: RunKind, worker_id: usize, total_workers: usize) {
    std::fs::create_dir_all(kind.dir()).expect("cannot create output dir");

    let shard_name = kind.shard_name(worker_id);
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&shard_name)
            .expect("cannot open shard"),
    );
    if writer.get_ref().metadata().unwrap().len() == 0 {
        writeln!(writer, "secret_idx,entropy,moves_remaining").unwrap();
    }

    let mut solver = WordleSolver::intialise(
        &"./words_5_letters.txt".to_owned(),
        Policy::MaximizeEntropy,
        Vec::new(),
    )
    .unwrap();

    let max_secrets = 1_500.min(solver.dictionary.len());
    for secret_idx in (0..max_secrets).filter(|i| i % total_workers == worker_id) {
        let secret = solver.dictionary[secret_idx];
        solver.reset();
        let mut entropies = Vec::new();
        let mut guesses = 0;

        loop {
            entropies.push((solver.current_possibilities.len() as f64).log2());
            solver.step();
            guesses += 1;

            let guess = solver.current_guess.unwrap();
            let feedback = guess.match_result(&secret);
            solver.guess(|_| feedback);

            if solver.current_possibilities.len() == 1 || guesses == 6 {
                break;
            }
        }
        let total = guesses as i32;
        for (step, &e) in entropies.iter().enumerate() {
            writeln!(writer, "{},{},{}", secret_idx, e, total - step as i32).unwrap();
        }
        writer.flush().unwrap();
    }
}

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Train {
        #[arg(short, long, default_value_t = 0)]
        workers: usize,
    },
    Test {
        #[arg(short, long, default_value_t = 0)]
        workers: usize,
    },
    TrainWorker {
        worker_id: usize,
        total_workers: usize,
    },
    TestWorker {
        worker_id: usize,
        total_workers: usize,
    },
    Play,
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Train { workers } => spawn_workers(workers, RunKind::Train),
        Cmd::Test { workers } => spawn_workers(workers, RunKind::Test),
        Cmd::TrainWorker {
            worker_id,
            total_workers,
        } => run_generic_worker(RunKind::Train, worker_id, total_workers),
        Cmd::TestWorker {
            worker_id,
            total_workers,
        } => run_generic_worker(RunKind::Test, worker_id, total_workers),
        Cmd::Play => interactive_play(),
    }
}
