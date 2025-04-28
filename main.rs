use gnuplot::FillPatternType::Pattern0;
use gnuplot::{AxesCommon, ColorType, Figure};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

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

// Open 5 letter words dictionary
fn open_dictionary<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    let mut success = false;
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let words: Vec<String> = reader.lines().filter_map(Result::ok).collect();

    Ok(words)
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
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
        let mut likelihood: f32 = 1.0;

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

    pub fn match_result(&self, other: &WordEncoding) -> [MatchKind; 5] {
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

    pub fn match_results(&self, other: &Vec<WordEncoding>) -> Vec<[MatchKind; 5]> {
        let mut results: Vec<[MatchKind; 5]> = Vec::new();

        for word in other {
            results.push(self.match_result(word));
        }

        results
    }
}

fn compute_and_save_word_encodings(words: &Vec<String>, save_path: &str) -> io::Result<()> {
    let mut encodings: Vec<WordEncoding> = Vec::new();
    // Compute encoding for each word
    for word in words {
        encodings.push(WordEncoding::from_string(word));
    }
    let bytes = bincode::serialize(&encodings)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Serialization error: {}", e)))?;

    let mut file = File::create(save_path)?;

    file.write_all(&bytes)?;

    Ok(())
}

fn load_word_encodings(path: &str) -> io::Result<Vec<WordEncoding>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();

    file.read_to_end(&mut buffer)?;

    let value = bincode::deserialize(&buffer).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Deserialization error: {}", e),
        )
    })?;

    Ok(value)
}

// Compute the 'match pattern' probability distribution (pd), of a given word over the possibility
fn compute_match_pattern_pd(match_results: &Vec<[MatchKind; 5]>) -> [f32; 243] {
    let mut match_pattern_pd: [f32; 243] = [0.0; 243];

    for match_result in match_results {
        // Compute the index
        let mut index: usize = 0;

        for i in 0..5 {
            match match_result[i] {
                MatchKind::NoMatch => index += 0 * (3 as usize).pow(i as u32),
                MatchKind::Partial => index += 1 * (3 as usize).pow(i as u32),
                MatchKind::Match => index += 2 * (3 as usize).pow(i as u32),
            }
        }

        match_pattern_pd[index] += 1.0;
    }

    // Normalise
    let sum: f32 = match_pattern_pd.iter().sum();

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

fn generate_word_encodings() {
    let words = open_dictionary("./words_5_letters.txt").unwrap_or_else(|err| {
        println!("Error openning dictionary: {}", err);
        std::process::exit(-1);
    });

    let result = compute_and_save_word_encodings(&words, "./word_encodings.bin");

    match result {
        Ok(()) => println!("Encodings saved to ./word_encodings.bin"),
        Err(err) => {
            println!("Failed to create encodings: {}", err);
            std::process::exit(-1);
        }
    }
}

fn plot_match_pattern_pd<const N: usize>(pd: [f32; N]) {
    // Plot using GNU plot

    // Generate x-values: 0, 1, 2, 3 (indices)
    let x: Vec<_> = (0..pd.len()).map(|i| i as f32).collect();

    let mut fg = Figure::new();
    {
        let axes = fg.axes2d();
        axes.set_title("Probability Distribution", &[]).boxes(
            &x,
            &pd,
            &[
                gnuplot::Caption("Probability"),
                gnuplot::Color(ColorType::RGBString("blue")),
            ],
        );
    }

    fg.show().unwrap();
}

fn main() {
    // Open the dictionary and compute encoding
    // generate_word_encodings();

    // Load encodings
    {
        let mut encodings = load_word_encodings("./word_encodings.bin").unwrap_or_else(|err| {
            println!("Error loading word encodings: {}", err);
            std::process::exit(-1);
        });
        println!("Number of possibilities: {}", encodings.len());

        while encodings.len() > 1 {
            let mut guess: Option<WordEncoding> = None;
            let mut guess_entropy: f32 = 0.0;
            let mut guess_match_result: Vec<[MatchKind; 5]> = Vec::new();
            // Compute the entropy of each word against the current set of encodings
            for encoding in encodings.iter() {
                let match_results = encoding.match_results(&encodings);
                let match_pattern_pd = compute_match_pattern_pd(&match_results);
                let entropy = compute_entropy(match_pattern_pd);
                // plot_match_pattern_pd(match_pattern_pd);
                if entropy > guess_entropy {
                    guess = Some(*encoding);
                    guess_entropy = entropy;
                    guess_match_result = match_results;
                }
            }

            if let Some(some_guess) = &guess {
                println!(
                    "Guessing with {} at an expected entropy of {}",
                    some_guess.to_string(),
                    guess_entropy
                );

                let actual_match = some_guess.match_result(&WordEncoding::from_string("WEARY"));

                let keep_indices: Vec<usize> = guess_match_result
                    .iter()
                    .enumerate()
                    .filter(|(_, val)| **val == actual_match)
                    .map(|(index, _)| index)
                    .collect();

                let keep_encodings: Vec<_> = keep_indices.iter().map(|&i| encodings[i]).collect();

                encodings = keep_encodings;

                println!("Remaining possibilities: {}", encodings.len());
            } else {
                println!("Cannot determine guess");
                break;
            }
        }

        println!("Final/Possible answer(s): ");
        for encoding in encodings {
            println!("{}", encoding.to_string());
        }
    }

    /*
    println!("Wordle Solver!\n");
    let solver = WordleSolver {};
    let result = wordle_solve(solver, check_function);

    println!("{}", result);
    */
}
