static TRAIN: [[f32; 2]; 5] = [
    [0f32, 0.5f32],
    [1f32, 2.5f32],
    [2f32, 4.5f32],
    [3f32, 6.5f32],
    [4f32, 8.5f32],
];
fn loss(w: f32, b: f32, sample: &[[f32; 2]]) -> f32 {
    let mut res = 0f32;
    for [x, y] in sample {
        let d = y - x * w - b;
        res += d * d;
    }
    res / sample.len() as f32
}

fn main() {
    let mut w = rand::random_range(0.0..=10.0);
    let mut b = 0f32;
    const EPS: f32 = 1e-4;
    let (lr, limit, times, n) = parse_args();
    let sample = get_sample(n);
    for _ in 0..times {
        let d1 = (loss(w + EPS, b, &sample) - loss(w, b, &sample)) / EPS;
        let d2 = (loss(w, b + EPS, &sample) - loss(w, b, &sample)) / EPS;
        if d1.abs() <= limit && d2.abs() <= limit {
            break;
        }
        w -= lr * d1;
        b -= lr * d2;
        println!("DEBUG:w={w} b={b} with loss: {:>5}", loss(w, b, &sample))
    }
    println!("w={w},b={b}")
}
fn get_sample(n: usize) -> Vec<[f32; 2]> {
    let mut ret = Vec::with_capacity(n);
    for i in 0..n {
        let i = i as f32;
        ret.push([i, 2 as f32 * i + 0.5]);
    }
    ret
}
fn parse_args() -> (f32, f32, u32, usize) {
    let mut args = std::env::args();
    args.next();
    let buf = args.next();
    let lr: f32 = buf
        .unwrap_or("0.01".into())
        .trim()
        .parse()
        .expect("invalid number");
    let buf = args.next();
    let limit: f32 = buf
        .unwrap_or("0.1".into())
        .trim()
        .parse()
        .expect("invalid number");
    let buf = args.next();
    let times: u32 = buf
        .unwrap_or("1000".into())
        .trim()
        .parse()
        .expect("invalid number");
    let buf = args.next();
    let n: usize = buf
        .unwrap_or("10".into())
        .trim()
        .parse()
        .expect("invalid number");
    (lr, limit, times, n)
}
#[cfg(test)]
mod tests {
    use super::TRAIN;
    #[test]
    fn test() {
        print!("{TRAIN:?}")
    }
}
