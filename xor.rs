pub struct XorModel<F> 
where F:Fn(f32) -> f32 {
    w_nand_1: f32,
    w_nand_2: f32,
    b_nand: f32,
    w_or_1: f32,
    w_or_2: f32,
    b_or: f32,
    sigmoid:F
}
impl<F> XorModel<F> 
where F:Fn(f32) -> f32{
    pub fn new(sigmoid:F) -> Self {
        Self {
            w_nand_1: rand::random(),
            w_nand_2: rand::random(),
            b_nand: rand::random(),
            w_or_1: rand::random(),
            w_or_2: rand::random(),
            b_or: rand::random(),
            sigmoid
        }
    }
    fn nand(&self, x1: f32, x2: f32) -> f32 {
        (self.sigmoid)(self.w_nand_1 * x1 + self.w_nand_2 * x2 + self.b_nand)
    }

    fn or(&self, x1: f32, x2: f32) -> f32 {
        (self.sigmoid)(self.w_or_1 * x1 + self.w_or_2 * x2 + self.b_or)
    }
    pub fn train(
        &mut self, 
        lr: f32, 
        times: u32,
        mut loss_exp: impl FnMut(f32, f32, f32, &[[f32; 3]; 4]) -> f32,
    ) {
        const EPS: f32 = 1e-3;
        let nand_sample = [
            [1f32, 0f32, 0f32],
            [0f32, 0f32, 0f32],
            [0f32, 1f32, 1f32],
            [1f32, 1f32, 0f32],
        ];

        let or_sample = [
            [1f32, 0f32, 1f32],
            [0f32, 0f32, 0f32],
            [0f32, 1f32, 1f32],
            [1f32, 1f32, 1f32],
        ];
        for _ in 0..times {
            let d = loss_exp(self.w_nand_1, self.w_nand_2, self.b_nand, &nand_sample);
            let d1 = (loss_exp(
                self.w_nand_1 + EPS,
                self.w_nand_2,
                self.b_nand,
                &nand_sample,
            ) - d)
                / EPS;
            let d2 = (loss_exp(
                self.w_nand_1,
                self.w_nand_2 + EPS,
                self.b_nand,
                &nand_sample,
            ) - d)
                / EPS;
            let d3 = (loss_exp(
                self.w_nand_1,
                self.w_nand_2,
                self.b_nand + EPS,
                &nand_sample,
            ) - d)
                / EPS;
            self.w_nand_1 -= lr * d1;
            self.w_nand_2 -= lr * d2;
            self.b_nand -= lr * d3;
            print!("DEBUG: nand:loss={d}");
            let d = loss_exp(self.w_or_1, self.w_or_2, self.b_or, &or_sample);
            let d1 = (loss_exp(self.w_or_1 + EPS, self.w_or_2, self.b_or, &or_sample) - d) / EPS;
            let d2 = (loss_exp(self.w_or_1, self.w_or_2 + EPS, self.b_or, &or_sample) - d) / EPS;
            let d3 = (loss_exp(self.w_or_1, self.w_or_2, self.b_or + EPS, &or_sample) - d) / EPS;
            self.w_or_1 -= lr * d1;
            self.w_or_2 -= lr * d2;
            self.b_or -= lr * d3;
            print!("\t or:loss={d}\n");
        }

        for i in 0..=1 {
            for j in 0..=1 {
                let a = self.nand(i as f32, j as f32);
                let b = self.nand(j as f32, i as f32);
                let c = self.or(a, b);
                println!("i={i},j={j},a={a},b={b},c={c}");
            }
        }
    }
}

