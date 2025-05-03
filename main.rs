use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

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

#[derive(Clone, Copy, Debug)]
struct WordEncoding {
    positions: [char; 5],  // Encode symbol position
    frequencies: [u8; 26], // Encode symbol frequency
    likelihood: f32,       // Likelyhood of the word
}

impl WordEncoding {
    pub fn to_string(&self) -> String {
        self.positions.iter().collect()
    }

    pub fn from_string(word: &str) -> WordEncoding {
        let mut positions = ['A'; 5];
        let mut frequencies = [0u8; 26];
        let likelihood: f32 = 1.0;

        for (i, c) in word.chars().enumerate() {
            positions[i] = c.to_ascii_uppercase();
            let idx = (c.to_ascii_uppercase() as u8 - b'A') as usize;
            frequencies[idx] += 1;
        }

        WordEncoding {
            positions,
            frequencies,
            likelihood,
        }
    }

    pub fn match_result(&self, other: &WordEncoding) -> MatchResult {
        let mut result = [
            MatchKind::NoMatch,
            MatchKind::NoMatch,
            MatchKind::NoMatch,
            MatchKind::NoMatch,
            MatchKind::NoMatch,
        ];

        for (i, letter) in self.positions.iter().enumerate() {
            if other.positions[i] == *letter {
                result[i] = MatchKind::Match;
            } else {
                let idx = (letter.to_ascii_uppercase() as u8 - b'A') as usize;
                if other.frequencies[idx] > 0 {
                    result[i] = MatchKind::Partial;
                }
            }
        }

        result
    }

    pub fn match_results(
        &self,
        other: &Vec<WordEncoding>,
        indices: &Vec<usize>,
    ) -> Vec<(MatchResult, f32)> {
        let mut results: Vec<(MatchResult, f32)> = Vec::new();

        for i in indices {
            results.push((self.match_result(&other[*i]), other[*i].likelihood));
        }

        results
    }
}

struct WordleSolver {
    dictionary: Vec<WordEncoding>,
    current_possibilities: Vec<usize>,
    current_guess: Option<WordEncoding>,
    current_guess_entropy: f32,
    current_guess_match_result: Option<Vec<(MatchResult, f32)>>,
    current_guess_match_pattern_pd: Option<[f32; 243]>,
    current_expected_score: f32,
}

impl WordleSolver {
    pub fn intialise(dictionary_path: &String) -> Result<WordleSolver, String> {
        let solver: WordleSolver;

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
            current_possibilities: (0..dictionary_len).collect(),
            current_guess: None,
            current_guess_entropy: 0.0,
            current_guess_match_result: None,
            current_guess_match_pattern_pd: None,
            current_expected_score: WordleSolver::compute_expected_score(
                (dictionary_len as f32).log2(),
            ),
        };

        Ok(solver)
    }

    pub fn reset(&mut self) {
        // Reset values
        self.current_guess = None;
        self.current_guess_entropy = 0.0;
        self.current_guess_match_result = None;
        self.current_guess_match_pattern_pd = None;

        // Reset possibilties
        self.current_possibilities = (0..self.dictionary.len()).collect();
    }

    pub fn guess<CheckFunction>(&mut self, callback: CheckFunction)
    where
        CheckFunction: Fn(&WordEncoding) -> MatchResult,
    {
        if let Some(some_guess) = &self.current_guess {
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
        }
    }

    pub fn step(&mut self) {
        self.current_guess = None;
        self.current_guess_entropy = 0.0;
        self.current_guess_match_result = None;
        self.current_guess_match_pattern_pd = None;

        // Calculate entropy of every possibilities
        for guess in self.dictionary.iter() {
            let match_results = guess.match_results(&self.dictionary, &self.current_possibilities);
            let match_pattern_pd = WordleSolver::compute_match_pattern_pd(&match_results);
            let entropy = WordleSolver::compute_entropy(match_pattern_pd);

            // TODO: Mimise expected score rather than entropy
            if entropy > self.current_guess_entropy {
                self.current_guess = Some(*guess);
                self.current_guess_entropy = entropy;
                self.current_guess_match_result = Some(match_results);
                self.current_guess_match_pattern_pd = Some(match_pattern_pd);
            }
        }
    }

    fn compute_word_encodings(words: &Vec<String>) -> Vec<WordEncoding> {
        let mut encodings: Vec<WordEncoding> = Vec::new();
        let parametric_sigmoid = |x: f32, midpoint: f32, steepness: f32| -> f32 {
            1.0 / (1.0 + (-steepness * (x - midpoint)).exp())
        };

        // Compute encoding for each word
        for (i, word) in words.iter().enumerate() {
            let mut encoding = WordEncoding::from_string(word);
            let x = (words.len() - i - 1) as f32;
            encoding.likelihood = parametric_sigmoid(x, (words.len() as f32) / 2.0, 0.02);
            println!("{}: {}", encoding.to_string(), encoding.likelihood);
            encodings.push(encoding);
        }

        encodings
    }

    // Compute the 'match pattern' probability distribution (pd), of a given word over the possibility
    fn compute_match_pattern_pd(match_results: &Vec<(MatchResult, f32)>) -> [f32; 243] {
        let mut sum: f32 = 0.0;
        let mut match_pattern_pd: [f32; 243] = [0.0; 243];

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

    fn compute_entropy<const N: usize>(pd: [f32; N]) -> f32 {
        let mut entropy: f32 = 0.0;
        for probabilty in pd.iter() {
            if *probabilty > 0.0 {
                entropy += -1.0 * (*probabilty) * (*probabilty).log2();
            }
        }

        entropy
    }

    fn compute_expected_score(entropy: f32) -> f32 {
        /// TODO: Write function to determine the expected score given the amount of entropy
        0.0
    }
}

fn main() {
    let mut solver = match WordleSolver::intialise(&"./words_5_letters.txt".to_string()) {
        Ok(solver) => solver,
        Err(error) => {
            eprintln!("failed to initialize WordleSolver: {}", error);
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
            "Guess: {}, Expected ΔEntropy: {}, Remaining Possibilities: {}",
            guess.to_string(),
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

        let actual_entropy: f32 = f32::log2(initial_possibilities as f32)
            - f32::log2(solver.current_possibilities.len() as f32);

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
