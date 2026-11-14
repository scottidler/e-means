use super::*;

// Golden vectors for seed 1234. They pin the whole contract (SplitMix64 key, stream ids, ChaCha8 bit
// stream, 53-bit uniform, Lemire bounded ints, Fisher-Yates): a change to any of them fails here first.

#[test]
fn golden_key_expansion_is_splitmix64() {
    // Cross-checked against a Python transcription of SplitMix64 (Steele, Lea and Flood 2014).
    let mut state = 1234u64;
    let words: Vec<u64> = (0..4).map(|_| splitmix64(&mut state)).collect();
    assert_eq!(
        words,
        [
            13478418381427711195,
            10936887474700444964,
            3728693401281897946,
            5648149391703318579
        ]
    );
    let key = expand_seed(1234);
    assert_eq!(key[..8], 13478418381427711195u64.to_le_bytes());
    assert_eq!(key[24..], 5648149391703318579u64.to_le_bytes());
}

#[test]
fn golden_first_8_u64_of_the_revival_stream() {
    let mut stream = Stream::new(1234, StreamId::Revival);
    let words: Vec<u64> = (0..8).map(|_| stream.next_u64()).collect();
    assert_eq!(
        words,
        [
            7839892025930919283,
            4898396843259606927,
            9463475350526664508,
            9108384216797613808,
            15029637854104988123,
            15329761231136882789,
            4486195400716396707,
            7132333170260767736
        ]
    );
}

#[test]
fn golden_first_4_uniforms_of_the_revival_stream() {
    let mut stream = Stream::new(1234, StreamId::Revival);
    let uniforms: Vec<u64> = (0..4).map(|_| stream.uniform().to_bits()).collect();
    let want: Vec<u64> = [
        0.4250013983283042f64,
        0.26554262495791003,
        0.5130160267152014,
        0.49376649778423254,
    ]
    .iter()
    .map(|u| u.to_bits())
    .collect();
    assert_eq!(uniforms, want);
}

#[test]
fn golden_permutation_of_10_on_pass_0() {
    // Cross-checked against a Python transcription of Lemire + Fisher-Yates over the same u64 stream.
    let mut stream = Stream::new(1234, StreamId::Pass(0));
    assert_eq!(stream.permutation(10), [5, 3, 7, 1, 2, 8, 0, 6, 9, 4]);
}

#[test]
fn golden_streams_are_distinct_per_use() {
    let first = |id| Stream::new(1234, id).next_u64();
    assert_eq!(first(StreamId::Pass(0)), 7751969689126549159);
    assert_eq!(first(StreamId::Init), 13412373498528894678);
    assert_eq!(first(StreamId::Revival), 7839892025930919283);
    assert_eq!(first(StreamId::Subset), 17003396118023392089);
    assert_ne!(first(StreamId::Pass(1)), first(StreamId::Pass(0)));
    assert_ne!(
        Stream::new(1235, StreamId::Revival).next_u64(),
        first(StreamId::Revival)
    );
}

#[test]
fn uniform_is_in_the_unit_interval() {
    let mut stream = Stream::new(7, StreamId::Revival);
    for _ in 0..10_000 {
        let u = stream.uniform();
        assert!((0.0..1.0).contains(&u), "{u}");
    }
}

#[test]
fn below_stays_in_range_and_reaches_every_value() {
    let mut stream = Stream::new(7, StreamId::Pass(3));
    let mut seen = [0usize; 3];
    for _ in 0..3_000 {
        seen[stream.below(3) as usize] += 1;
    }
    assert!(seen.iter().all(|&c| c > 800), "{seen:?}");
    assert_eq!(stream.below(1), 0);
}

#[test]
fn below_rejects_the_biased_low_products() {
    // bound = 2^63 + 1 rejects about half of all draws: the rejection loop runs and stays in range.
    let bound = (1u64 << 63) + 1;
    let mut stream = Stream::new(7, StreamId::Pass(4));
    for _ in 0..1_000 {
        assert!(stream.below(bound) < bound);
    }
    let draws = stream.rng.get_word_pos() / 2;
    assert!(draws > 1_000, "rejection never drew a second number: {draws} draws");
}

#[test]
#[should_panic(expected = "positive bound")]
fn below_zero_panics() {
    Stream::new(1, StreamId::Pass(0)).below(0);
}

#[test]
fn permutation_is_a_seeded_permutation() {
    assert!(Stream::new(1, StreamId::Pass(0)).permutation(0).is_empty());
    assert_eq!(Stream::new(1, StreamId::Pass(0)).permutation(1), [0]);
    let order = Stream::new(9, StreamId::Pass(2)).permutation(1000);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
    assert_eq!(order, Stream::new(9, StreamId::Pass(2)).permutation(1000));
    assert_ne!(order, Stream::new(10, StreamId::Pass(2)).permutation(1000));
}
