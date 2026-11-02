mod xor;
fn loss(w: f32, b: f32, sample: &[[f32; 2]]) -> f32 {
    let mut res = 0f32;
    for [x, y] in sample {
        let d = y - x * w - b;
        res += d * d;
    }
    res / sample.len() as f32
}
fn main() {
    let mut w = rand::random();
    let mut b = 0f32;
    const EPS: f32 = 1e-3;
    let (lr, times, n) = parse_args();
    let sample = get_sample(n);
    for _ in 0..times {
        let d = loss(w, b, &sample);
        let d1 = (loss(w + EPS, b, &sample) - d) / EPS;
        let d2 = (loss(w, b + EPS, &sample) - d) / EPS;
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

fn parse_args() -> (f32, u32, usize) {
    let mut args = std::env::args();
    args.next();
    let buf = args.next();
    let lr: f32 = buf
        .unwrap_or("0.01".into())
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
    (lr, times, n)
}
#[cfg(test)]
mod tests {
    use super::xor;
    use std::io::Write;
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-x))
    }
    fn loss<const N: usize>(w1: f32, w2: f32, b: f32, sample: &[[f32; 3]; N]) -> f32 {
        let mut res = 0f32;
        for [x1, x2, y] in sample {
            let d = y - sigmoid(x1 * w1 + x2 * w2 + b);
            res += d * d;
        }
        res / sample.len() as f32
    }
    /// 二元交叉熵：-mean(y * ln(p) + (1 - y) * ln(1 - p))，p = sigmoid(z)。
    fn loss_exp<const N: usize>(w1: f32, w2: f32, b: f32, sample: &[[f32; 3]; N]) -> f32 {
        let mut res = 0.0_f32;
        let ln = f32::ln;
        for [x1, x2, y] in sample {
            let z = x1 * w1 + x2 * w2 + b;
            let p = sigmoid(z).clamp(f32::EPSILON, 1.0 - f32::EPSILON);
            res -= y * ln(p) + (1.0 - y) * ln(1.0 - p);
        }
        res / N as f32
    }
    #[test]
    fn test_and() {
        do_test_and(loss);
    }
    #[test]
    fn test_and_exp() {
        do_test_and(loss_exp);
    }
    /// test the operation `AND`,e.g. 0 & 1= 0
    fn do_test_and(mut loss: impl FnMut(f32, f32, f32, &[[f32; 3]; 4]) -> f32) {
        let mut w1 = rand::random();
        let mut w2 = rand::random();
        let mut b = rand::random();
        const EPS: f32 = 1e-3;
        let (lr, times) = (1e-4 as f32, 10000000);
        let sample = [
            [1f32, 0f32, 0f32],
            [0f32, 0f32, 0f32],
            [0f32, 1f32, 0f32],
            [1f32, 1f32, 1f32],
        ];
        let mut handle = std::io::BufWriter::new(std::fs::File::create("log").unwrap());
        for _ in 0..times {
            let d = loss(w1, w2, b, &sample);
            let d1 = (loss(w1 + EPS, w2, b, &sample) - d) / EPS;
            let d2 = (loss(w1, w2 + EPS, b, &sample) - d) / EPS;
            let d3 = (loss(w1, w2, b + EPS, &sample) - d) / EPS;
            w1 -= lr * d1;
            w2 -= lr * d2;
            b -= lr * d3;
            handle
                .write(format!("DEBUG:w1={w1} w2={w2} b={b} with loss: {:>10}\n", d).as_bytes())
                .expect("ERROR: can't write to the file");
        }
        handle
            .write(format!("w1={w1},w2={w2},b={b}\n").as_bytes())
            .expect("ERROR: can't write to the file");
        for i in 0..=1 {
            for j in 0..=1 {
                handle
                    .write(
                        format!(
                            "i={i},j={j},output={res}\n",
                            res = sigmoid(w1 * i as f32 + w2 * j as f32 + b)
                        )
                        .as_bytes(),
                    )
                    .expect("ERROR: can't write to the file");
            }
        }
    }
    #[test]
    fn test_xor() {
        xor::XorModel::new(sigmoid).train(1e-3, 500000, loss_exp);
    }
}
