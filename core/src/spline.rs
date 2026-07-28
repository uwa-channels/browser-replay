//! Not-a-knot cubic spline interpolation, matching the reference
//! `interp1(x, v, xq, 'spline')` (interior/extrapolated evaluation) and
//! `interp1(x, v, xq, 'spline', fill)` (explicit fill value outside the
//! domain of `x`, as used by the reference channel replay implementation
//! for both the impulse response tap-trajectory interpolation and the
//! Doppler/delay-drift time warp). Ported from a not-a-knot solver,
//! generalized to work over real (`f64`) or complex (`Complex64`) values,
//! since channel taps are complex baseband gains while the drift-warp
//! values are real.

use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

/// A value type a spline can hold: a vector space over `f64` scalars.
/// Implemented for `f64` and, via `num_complex`'s generic scalar ops, for
/// `Complex64`.
pub trait SplineValue:
    Copy
    + Zero
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
}
impl<T> SplineValue for T where
    T: Copy
        + Zero
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<f64, Output = T>
        + Div<f64, Output = T>
{
}

/// Thomas algorithm for a tridiagonal system with real sub/main/super
/// diagonals and a (possibly complex) right-hand side.
fn thomas<T: SplineValue>(sub: &[f64], dia: &[f64], sup: &[f64], rhs: &[T]) -> Vec<T> {
    let m = dia.len();
    let mut cp = vec![0.0_f64; m];
    let mut dp = vec![T::zero(); m];
    cp[0] = sup[0] / dia[0];
    dp[0] = rhs[0] / dia[0];
    for i in 1..m {
        let denom = dia[i] - sub[i] * cp[i - 1];
        cp[i] = sup[i] / denom;
        dp[i] = (rhs[i] - dp[i - 1] * sub[i]) / denom;
    }
    let mut sol = vec![T::zero(); m];
    sol[m - 1] = dp[m - 1];
    for i in (0..m - 1).rev() {
        sol[i] = dp[i] - sol[i + 1] * cp[i];
    }
    sol
}

/// Largest `k` with `x[k] <= xq`, clamped to `0..x.len()-2` so the same index
/// is usable both for interior interpolation and for extrapolation via the
/// boundary segment's cubic piece.
fn locate(x: &[f64], xq: f64) -> usize {
    let n = x.len();
    if xq <= x[0] {
        return 0;
    }
    if xq >= x[n - 1] {
        return n - 2;
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// A not-a-knot cubic spline through `(x[i], v[i])`, precomputed once and
/// evaluated at arbitrary query points.
pub struct CubicSpline<T: SplineValue> {
    x: Vec<f64>,
    v: Vec<T>,
    /// Second derivatives at each knot.
    m: Vec<T>,
}

impl<T: SplineValue> CubicSpline<T> {
    pub fn new(x: &[f64], v: &[T]) -> Self {
        let n = x.len();
        assert_eq!(x.len(), v.len(), "x and v must have the same length");
        assert!(n >= 2, "CubicSpline needs at least 2 points");

        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let dd: Vec<T> = (0..n - 1).map(|i| (v[i + 1] - v[i]) / h[i]).collect();

        let m = if n == 2 {
            vec![T::zero(); 2]
        } else if n == 3 {
            let val = ((dd[1] - dd[0]) / (x[2] - x[0])) * 2.0;
            vec![val; 3]
        } else {
            let m_count = n - 2;
            let mut sub = vec![0.0_f64; m_count];
            let mut dia = vec![0.0_f64; m_count];
            let mut sup = vec![0.0_f64; m_count];
            let mut rhs = vec![T::zero(); m_count];
            for (u, (sub_u, (dia_u, sup_u))) in sub
                .iter_mut()
                .zip(dia.iter_mut().zip(sup.iter_mut()))
                .enumerate()
            {
                let i = u + 1;
                *sub_u = h[i - 1];
                *dia_u = 2.0 * (h[i - 1] + h[i]);
                *sup_u = h[i];
                rhs[u] = (dd[i] - dd[i - 1]) * 6.0;
            }
            let (h1, h2) = (h[0], h[1]);
            dia[0] = 3.0 * h1 + h1 * h1 / h2 + 2.0 * h2;
            sup[0] = (h2 * h2 - h1 * h1) / h2;
            sub[0] = 0.0;
            let (hm2, hm1) = (h[n - 3], h[n - 2]);
            sub[m_count - 1] = (hm2 * hm2 - hm1 * hm1) / hm2;
            dia[m_count - 1] = 2.0 * hm2 + 3.0 * hm1 + hm1 * hm1 / hm2;
            sup[m_count - 1] = 0.0;

            let mi = thomas(&sub, &dia, &sup, &rhs);
            let mut m = vec![T::zero(); n];
            m[1..n - 1].copy_from_slice(&mi);
            m[0] = m[1] - (m[2] - m[1]) * (h1 / h2);
            m[n - 1] = m[n - 2] + (m[n - 2] - m[n - 3]) * (hm1 / hm2);
            m
        };

        CubicSpline {
            x: x.to_vec(),
            v: v.to_vec(),
            m,
        }
    }

    /// Evaluate at `xq`. Outside `[x[0], x[n-1]]` this extrapolates using the
    /// boundary segment's cubic piece — matching `interp1(x,v,xq,'spline')`
    /// with `'extrap'` (or the bare `spline()` function).
    pub fn eval_extrap(&self, xq: f64) -> T {
        let k = locate(&self.x, xq);
        let hi = self.x[k + 1] - self.x[k];
        let a = self.x[k + 1] - xq;
        let b = xq - self.x[k];
        let (mk, mk1) = (self.m[k], self.m[k + 1]);
        let (vk, vk1) = (self.v[k], self.v[k + 1]);
        mk * (a * a * a / (6.0 * hi))
            + mk1 * (b * b * b / (6.0 * hi))
            + (vk / hi - mk * (hi / 6.0)) * a
            + (vk1 / hi - mk1 * (hi / 6.0)) * b
    }

    /// Evaluate at `xq`, returning `fill` for any query outside
    /// `[x[0], x[n-1]]` instead of extrapolating — matches
    /// `interp1(x, v, xq, 'spline', fill)`, the form used throughout
    /// `replay.m`/`replay.py`.
    pub fn eval_fill(&self, xq: f64, fill: T) -> T {
        if xq < self.x[0] || xq > self.x[self.x.len() - 1] {
            fill
        } else {
            self.eval_extrap(xq)
        }
    }

    pub fn eval_many_fill(&self, xs: &[f64], fill: T) -> Vec<T> {
        xs.iter().map(|&xq| self.eval_fill(xq, fill)).collect()
    }

    pub fn eval_many_extrap(&self, xs: &[f64]) -> Vec<T> {
        xs.iter().map(|&xq| self.eval_extrap(xq)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn interpolates_exactly_at_knots() {
        let x = vec![0.0, 1.0, 2.5, 4.0, 6.0];
        let v = vec![1.0, 3.0, 2.0, 5.0, 0.0];
        let s = CubicSpline::new(&x, &v);
        for i in 0..x.len() {
            let got = s.eval_extrap(x[i]);
            assert!((got - v[i]).abs() < 1e-9, "knot {i}: got {got}, want {}", v[i]);
        }
    }

    #[test]
    fn reproduces_cubic_polynomials_exactly() {
        // A not-a-knot cubic spline through samples of a cubic polynomial
        // must reproduce that polynomial exactly (classic spline property).
        let f = |x: f64| x * x * x - 2.0 * x * x + x - 3.0;
        let x: Vec<f64> = vec![0.0, 0.3, 1.0, 1.7, 2.2, 3.0, 4.1, 5.0];
        let v: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
        let s = CubicSpline::new(&x, &v);
        for q in [0.1, 0.5, 1.2, 1.9, 2.5, 3.3, 4.5] {
            let got = s.eval_extrap(q);
            let want = f(q);
            assert!(
                (got - want).abs() < 1e-8,
                "q={q}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn fill_value_outside_domain() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let v = vec![0.0, 1.0, 0.0, 1.0];
        let s = CubicSpline::new(&x, &v);
        assert_eq!(s.eval_fill(-1.0, 0.0), 0.0);
        assert_eq!(s.eval_fill(4.0, 0.0), 0.0);
        // Inside the domain, fill is irrelevant.
        assert!((s.eval_fill(0.0, 99.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn works_over_complex_values() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let v: Vec<Complex64> = x
            .iter()
            .map(|&xi| Complex64::new(xi.cos(), xi.sin()))
            .collect();
        let s = CubicSpline::new(&x, &v);
        for i in 0..x.len() {
            let got = s.eval_extrap(x[i]);
            assert!((got - v[i]).norm() < 1e-9);
        }
        let mid = s.eval_extrap(1.5);
        assert!(mid.norm() < 2.0); // sanity: interpolated point stays bounded
    }
}
