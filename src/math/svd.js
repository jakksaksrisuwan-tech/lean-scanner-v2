// One-sided Jacobi SVD for small matrices.
//
// Used only by the homography DLT, where A is 8x9. The python
// implementation calls numpy.linalg.svd which is LAPACK dgesdd
// (divide-and-conquer). We don't need that level of performance — Jacobi
// converges in O(n^2 * sweeps) and n=9 here, so it's instant.
//
// One-sided Jacobi computes Vt only; we don't need U or the singular
// values themselves for the DLT, just the right-singular vector for the
// smallest singular value. That's the last row of Vt.
//
// Reference: Golub & Van Loan, Matrix Computations §8.6
//            (one-sided orthogonal iteration).

/**
 * Compute the SVD of an m×n matrix A (m >= n).
 * Returns Vt — the n×n matrix of right singular vectors as rows.
 *
 * Implementation note: classic Jacobi rotates pairs of columns to
 * maximize their orthogonal separation. Converges quadratically per
 * sweep. We do at most 50 sweeps; in practice 4-8 is enough for 9x9.
 */
export function svdRightVectors(A, m, n) {
  // V starts as identity. We rotate pairs of columns to converge.
  const V = identity(n);
  const B = new Float64Array(A);  // working copy, will mutate

  const maxSweeps = 50;
  const tol = 1e-12;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        // Compute the 2x2 block B[:,p]' @ B[:,q] and the norms
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) {
          const bp = B[i * n + p];
          const bq = B[i * n + q];
          alpha += bp * bp;
          beta += bq * bq;
          gamma += bp * bq;
        }
        off += gamma * gamma;
        if (Math.abs(gamma) < tol * Math.sqrt(alpha * beta + 1e-300)) continue;

        // Compute the rotation that diagonalizes [alpha gamma; gamma beta]
        const tau = (beta - alpha) / (2 * gamma);
        const sign = tau >= 0 ? 1 : -1;
        const t = sign / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Apply rotation to B: B' = B * G(p,q,t)
        for (let i = 0; i < m; i++) {
          const bp = B[i * n + p];
          const bq = B[i * n + q];
          B[i * n + p] = c * bp - s * bq;
          B[i * n + q] = s * bp + c * bq;
        }
        // Accumulate into V: V' = V * G(p,q,t)
        for (let i = 0; i < n; i++) {
          const vp = V[i * n + p];
          const vq = V[i * n + q];
          V[i * n + p] = c * vp - s * vq;
          V[i * n + q] = s * vp + c * vq;
        }
      }
    }
    if (off < tol * tol) break;
  }

  // Vt: rows of V are the right singular vectors. Sort so smallest
  // singular value is the last row (matches np.linalg.svd's Vt layout).
  // To get the singular values we compute sqrt(diag(B'B)) before sorting,
  // then sort rows of V by corresponding sigma.
  const sigmas = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s2 = 0;
    for (let i = 0; i < m; i++) s2 += B[i * n + j] * B[i * n + j];
    sigmas[j] = Math.sqrt(s2);
  }
  // Sort indices by sigma ascending
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => sigmas[a] - sigmas[b]);
  const Vt = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const srcCol = idx[j];
    for (let i = 0; i < n; i++) {
      Vt[j * n + i] = V[i * n + srcCol];
    }
  }
  return Vt;
}

function identity(n) {
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) M[i * n + i] = 1;
  return M;
}