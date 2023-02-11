#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <unsupported/Eigen/FFT>
#include <cmath>




/*

Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd &A) {
  
  const int k = A.size();
  
  Eigen::MatrixXd Q(A); //Initialize Q with the values of A
  
  //Set the first vector
  Q.col(0).normalize();
  
  for(int j = 1; j < k; ++j) { 
    
    Q.col(j) = Q.col(j) - Q.leftCols(j) * 
                         (Q.leftCols(j).transpose() * A.col(j));
    
    //Get the numeric limit for the norm
    double eps = std::numeric_limits<double>::denorm_min();
    if(Q.col(j).norm() <= eps * A.col(j).norm()) {
      //This means that the columns of A are almost linearly dependent
      std::cout << "Gram-Schmidt failed." << std::endl;
      break;
    } else {
      Q.col(j).normalize(); 
    }
  }
  
  return Q;
}

bool testGramSchmidt(unsigned int n) {
  //First we construct the matrix A as specified
  Eigen::MatrixXd A(n, n);
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      A(i, j) = i + 2 * j;
    }
  }
  
  //Compute result of the gram schmidt function
  Eigen::MatrixXd Q = gram_schmidt(A);
  
  //Get the norm value
  double norm = (Q.transpose() * Q - Eigen::MatrixXd::Identity(n, n)).norm();
  
  //Get the numerical error limit
  double eps = std::numeric_limits<double>::denorm_min();
  
  if(norm < eps) {
    return true;
  } else {
    return false;
  }
}
*/
/*
void kron_mult_solution(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B, 
               const Eigen::VectorXd & x, Eigen::VectorXd & y) {
  assert(A.rows() == A.cols() && A.rows() == B.rows() && B.rows() == B.cols() &&
         "Matrices A and B must be square matrices with same size!");
  assert(x.size() == A.cols() * A.cols() &&
         "Vector x must have length A.cols()^2");
  const unsigned int n = A.rows();

  // Allocate space for output
  y = Eigen::VectorXd::Zero(n * n);

  // Loop over all segments of x ($\tilde{x}$)
  for (unsigned int j = 0; j < n; ++j) {
    // Reuse computation of z
    Eigen::VectorXd z = B * x.segment(j * n, n);
    // Loop over all segments of y
    for (unsigned int i = 0; i < n; ++i) {
      y.segment(i * n, n) += A(i, j) * z;
    }
  }
  // END
}


void kron_mult(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B, 
               const Eigen::VectorXd & x, Eigen::VectorXd & y) {
  //Check if dimensions match
  assert(A.rows() == B.cols() && "Matrices A and B must be compatible");
  assert(x.size() == B.rows() * A.rows() && "Vector x must have the correct length");
  
  const int A_rows = A.rows();
  const int A_cols = A.cols();
  
  const int B_rows = B.rows();
  const int B_cols = B.cols();
  
  //y will have size A.rows() * B.cols() <-- must be zero initialised (not Java)
  y = Eigen::VectorXd::Zero(A.rows() * B.cols());
  
  for(int i = 0; i < A_rows; ++i) {
    
    Eigen::VectorXd B_times_xSegment = B * x.segment(i * B_cols, B_cols); // x(index : index+l)
    
    for(int j = 0; j < A_cols; ++j) {
      //We use the structure seen in the exercise here and construct the sum column-wise
      y.segment(j * B_rows, B_rows) += A(j, i) * B_times_xSegment; 
    }
  }
}

void kron(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &C) {
  
  const int A_rows = A.rows();
  const int A_cols = A.cols();
  
  const int B_rows = B.rows();
  const int B_cols = B.cols();
  
  C = Eigen::MatrixXd(B_rows * A_rows, B_cols * A_cols);
  
  for(int i = 0; i < A_rows; ++i) {
    for(int j = 0; j < A_cols; ++j) {
      C.block(i * B_rows, j * B_cols, B_rows, B_cols) = A(i, j) * B;
    }
  }
}

void kron_reshape(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                  const Eigen::VectorXd &x, Eigen::VectorXd &y) {
  assert(A.rows() == A.cols() && A.rows() == B.rows() && B.rows() == B.cols() &&
         "Matrices A and B must be square matrices with same size!");
  const unsigned int n = A.rows();

  Eigen::MatrixXd temp = B * Eigen::MatrixXd::Map(x.data(), n, n) * A.transpose();
  y = Eigen::MatrixXd::Map(temp.data(), n * n, 1);
                               
}

void houserefl(const Eigen::VectorXd &v, Eigen::MatrixXd &Z) {
  
  const int n = v.size();
  
  Eigen::VectorXd w = v / v.norm();
  Eigen::VectorXd u = w;
  u(0) = u(0) + 1;
  Eigen::VectorXd q = u / u.norm();
  Eigen::MatrixXd X = Eigen::MatrixXd::Identity(n, n) - 2 * q * q.transpose();
  Z = X.rightCols(n - 1);
}

std::complex<double> myroot( std::complex<double> w ) {
    double x,y;
    double u = w.real();
    double v = w.imag();

    //START
    if(u >= 0) {
      x = (std::sqrt(u * u + v * v) + u) / 2.0;
      y = v / (2.0 * x);
    } else {
      y = (std::sqrt(u * u + v * v) - u) / 2.0;
      x = v / (2.0 * y);
    }
    
    //END

    return std::complex<double> (x,y);
}

void xmatmult(const Eigen::VectorXd& a, const Eigen::VectorXd& y, Eigen::VectorXd& x) {
  //Check that dimensions are compatible
  assert(a.size() == y.size() &&
         a.size() == x.size() && "A and x are not comatible.");
  
  const int n = a.size();
  
  //n / 2 rounds exactly as we want
  for(int i = 0; i < n / 2; ++i) {
    x(i) = a(i) * y(i) + a(n - i - 1) * y(n - i - 1);
    x(n - i - 1) = y(i);
  }
  
  //Treat odd case seperately
  if(n % 2 != 0) {
    x(n / 2) = a(n / 2) * y(n / 2);
  }
}


void multAx(const Eigen::VectorXd& a, const Eigen::VectorXd& b, 
            const Eigen::VectorXd& x, Eigen::VectorXd& y) {
  const int n = a.size() + 1;
  
  for(int i = 0; i < n; ++i) {
    //Initialize value to zero
    y(i) = 0;
    
    //First part
    if(i > 1) {
      y(i) += b(i - 2) * x(i - 2);
    } 
    
    //Second part
    y(i) += 2 * x(i);
    
    //Third part
    if(i < n - 1) {
      y(i) += a(i) * x(i + 1);
    }
  }
}

void solvelseAupper(const Eigen::VectorXd& a, const Eigen::VectorXd& r,
                    Eigen::VectorXd& x) {
  const int n = a.size() + 1; 

  assert(n == x.size() && x.size() == r.size() && "Input size does not match!");

  //Bottom equation first
  x(n - 1) = r(n - 1) / 2.0;
  
  //Solve the rest from bottom to top
  for(int i = n - 2; i >= 0; --i) {
    x(i) = (r(i) - a(i) * x(i + 1)) / 2.0;
  }
}

void solvelseA(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
               const Eigen::VectorXd& r , Eigen::VectorXd& x) {
  const int n = r.size();
  
  //Setup c and d, ensure c(0) = 0 and all d(i) = 2
  Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd d = (2 * Eigen::ArrayXd::Ones(n)).matrix();
  
  //Save old values of r in x -> they would be added anyway
  x = r;
  
  //Use new vector to store modified r_i
  Eigen::VectorXd r_new = r;
  
  //Compute values of c, d, r to do backwards subs. later
  for(int i = 0; i < n - 2; ++i) {
    c(i + 1) = -(b(i) / d(i)) * a(i);
    d(i + 1) = d(i + 1) - (c(i) / d(i)) * a(i);
    r_new(i + 1) = r_new(i + 1) - (c(i) / d(i)) * r_new(i);
    r_new(i + 2) = r_new(i + 2) - (b(i) / d(i)) * r_new(i);
  }
  
  //Do the backwards substitution
  x(n - 1) = r_new(n - 1) / d(n - 1);
  
  for(int i = n - 2; i >= 0; --i) {
    x(i) = (r_new(i) - a(i) * x(i + 1)) / d(i);
  }
}

Eigen::VectorXd solvelseASparse(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
                                const Eigen::VectorXd& r) {
  const int n = r.size();
  Eigen::SparseMatrix<double> A(n, n);
  A.reserve(3); //Reserve 3 non-zero entries per row
  for(int i = 0; i < n; ++i) {
    A.insert(i, i) = 2;
    
    if(i < n - 1) {
      A.insert(i, i + 1) = a(i);
    }
    
    if(i >= 2) {
      A.insert(i, i - 2) = b(i - 2);
    }
  }
  A.makeCompressed();
  
  //Use SparseLU solver (see lecture documet)
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(A); //Apparently a thing
  solver.compute(A);
  return solver.solve(r);
}

void solvelse(const Eigen::MatrixXd& R, const Eigen::VectorXd& v,
              const Eigen::VectorXd& u, const Eigen::VectorXd& bb,
              Eigen::VectorXd& x) {
                
  const int n = R.rows();
  double beta = bb(n);
  Eigen::VectorXd b = bb.head(n);
  
  //Check dimensions
  assert(n == R.cols() && n == v.size() && n == u.size() &&
         n + 1 == bb.size() && "dimensions must match");
         
  //Create TriangularView - arguments must match constness of parameters
  const Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Upper> triViewR =
        R.triangularView<Eigen::Upper>();
  
  //Solve first equation
  double uRb = -u.transpose() * triViewR.solve(b);
  double uRv = -u.transpose() * triViewR.solve(v); 
  
  double chi = (beta - uRb) / uRv;
  
  //Solve second equation
  Eigen::VectorXd z = triViewR.solve(b - v * chi);
  
  //Store the result in x
  x = Eigen::VectorXd(n + 1);
  x << z, chi;
}

void solvelseSol(const Eigen::MatrixXd& R, const Eigen::VectorXd& v,
              const Eigen::VectorXd& u, const Eigen::VectorXd& bb,
              Eigen::VectorXd& x) {
  // size of R, which is size of u, v, and size of bb is n+1
  const unsigned int n = R.rows();

  // TODO: (2-9.d)
  // i) Use assert() to check that R, v, u, bb all have the appropriate sizes.
  // ii) Use (3-9.b) to solve the LSE.
  // Hint: Use R.triangularView<Eigen::Upper>() to make use of the triangular
  // structure of R.
  // START
  assert(n == R.cols() && n == u.size() && n == v.size() &&
         n + 1 == bb.size() && "Size mismatch!");

  const Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Upper>& triR =
      R.triangularView<Eigen::Upper>();

  // $s$ is the Schur complement and, in this case, is a scalar
  // also $b_s$ is a scalar
  // $snv = s^{-1}$, $b_s$ as in lecture notes
  // $sinvbs = s^{-1}*b_s$
  const double sinv = -1. / u.dot(triR.solve(v));
  const double bs = (bb(n) - u.dot(triR.solve(bb.head(n))));
  const double sinvbs = sinv * bs;

  // Stack the vector $(z, \xi)^T =: x$
  x = Eigen::VectorXd::Zero(n + 1);
  x << triR.solve(bb.head(n) - v * sinvbs), sinvbs;
  // END
}

bool testSolveLSE(const Eigen::MatrixXd& R, const Eigen::VectorXd& v,
                  const Eigen::VectorXd& u, const Eigen::VectorXd& b,
                  Eigen::VectorXd& x) {
  
  const int n = R.size();
  //Check dimensions
  assert(n == R.cols() && n == u.size() && n == v.size() &&
         n + 1 == b.size() && "Size mismatch!");
         
  //Build matrix A
  Eigen::MatrixXd A(n + 1, n + 1);
  A << R, v, u.transpose(), 0;
  
  //Check with LU decomp first
  Eigen::FullPivLU<Eigen::MatrixXd> LU(A);
  Eigen::VectorXd solLU = LU.solve(b);
  
  //Check with our method
  Eigen::VectorXd solCustom;
  solvelse(R, v, u, b, solCustom);
  
  //Compare the norms - relative to the norm of b
  if((solLU - solCustom).norm() < 1e-8 * b.norm()) {
    return true;
  } else {
    return false;
  }
} 
void shift(Eigen::VectorXd & b) {
    int n = b.size();

    double temp = b(n-1);
    for(int k = n-2; k >= 0; --k) {
        b(k+1) = b(k);
    }
    b(0) = temp;
}
*/

/* @brief Compute $X = A^{-1}*[b_1,...,b_n],\; b_i = i$-th cyclic shift of $b$.
 * Function with naive implementation.
 * @param[in] A An $n \times n$ matrix
 * @param[in] b An $n$-dimensional vector
 * @param[out] X The $n \times n$ matrix $X = A^{-1}*[b_1,...,b_n]$
 */ /*
void solvpermbsol(const Eigen::MatrixXd & A, Eigen::VectorXd & b, Eigen::MatrixXd & X) {
    // Size of b, which is the size of A
    int n = b.size();
    assert( n == A.rows() && n == A.cols()
            && "Error: size mismatch!");
    X.resize(n,n);

    // For each loop iteration:
    // 1. solve the linear system $Ax = b$,
    // 2. store the result in a column of $X$
    // 3. and shift $b$ by one element for the next iteration.
    std::cout << "\n\nhello" << std::endl;
    for(int l = 0; l < n; ++l) {
        std::cout << b.transpose() << std::endl;
        X.col(l) = A.fullPivLu().solve(b);

        shift(b);
    }
    
    
}


void solvpermb(const Eigen::MatrixXd& A, Eigen::VectorXd& b, 
               Eigen::MatrixXd& X) {
                 
  const int n = b.size();               
  
  //Assert that dimensions match
  assert(n == A.rows() && n == A.cols() && "Dimension mismatch!");
  
  X = Eigen::MatrixXd(n, n);
  
  //Construct extended b - avoids actual shifting
  Eigen::VectorXd b_extended(2 * n);
  b_extended << b, b;

  Eigen::VectorXd b_permutated;
  Eigen::FullPivLU<Eigen::MatrixXd> LU(A);
  
  for(int i = 0; i < n; ++i) {
    b_permutated = b_extended.segment(n - i, n);
    X.col(i) = LU.solve(b_permutated);
  }
}

struct CRSMatrix {
  unsigned int m;
  unsigned int n;
  std::vector<double> val;
  std::vector<unsigned int> col_ind;
  std::vector<unsigned int> row_ptr;
};

bool GaussSeidelstep_crssol(const CRSMatrix &A, const Eigen::VectorXd &b,
                         Eigen::VectorXd &x) {
  assert(A.n == A.m && "Matrix must be square");
  assert(A.n == b.size() && "Vector b length mismatch");
  assert(A.n == x.size() && "Vector x length mismatch");

  // TODO: (2-18.b) Implement a single step of the Gauss-Seidel iteration with
  // the system matrix in CRS format.
  // START

  // Outer loop over rows of the matrix
  for (unsigned int i = 0; i < A.n; ++i) {
    double Aii = 0.0;
    double s = b[i];
    // Inner summation loop over non-zero entries of i-th row.
    // Skip diagonal entry in the summation and store is separately.
    for (unsigned int l = A.row_ptr[i]; l < A.row_ptr[i + 1]; ++l) {
      const unsigned int j = A.col_ind[l];
      if (j != i) {
        s -= A.val[l] * x[j];
      } else {
        // Fetch diagonal entry of A.
        Aii = A.val[l];
      }
    }
    if (Aii != 0.0)
      x[i] = s / Aii;
    else
      return false;
  }
  // END
  return true;
}


bool GaussSeidelstep_crs(const CRSMatrix& A, const Eigen::VectorXd& b,
                          Eigen::VectorXd& x) {
                            
  const int n = A.n; //Number of rows
  const int m = A.m; //Number of cols
  
  Eigen::VectorXd x_old;
  
  assert(n == m && m == b.size() && "Dimension mismatch!");
  
  for(int i = 0; i < n; ++i) {
    
    int colStart = A.row_ptr[i]; //StartIndex of element in current row in val
    int colEnd = A.row_ptr[i + 1]; //StartIndex of elements in next row in val
    double Aii = 0.0; //Diagonal entry default
    double x_next = b(i); //First term
    
    for(int j = colStart; j < colEnd; ++j) {
      int column = A.col_ind[j]; //Gives the column in A, i is the row in A
      
      if(i == column) { //Diagonal entry
        Aii = A.val[j];
      }
      //Account for first sum
      if(column <= i - 1) {
        x_next -= A.val[j] * x_next;
      }
      //Account for second sum
      if(column >= i + 1) {
        x_next -= A.val[j] * x(column);
      }
    }
    //Account for A not being well-defined
    if(Aii == 0.0) {
      return false;
    } else {
      x[i] = x_next / Aii;
    }
  } 
  return true;
}

bool GaussSeidel_iteration(const CRSMatrix& A, const Eigen::VectorXd& b,
                           Eigen::VectorXd& x, double atol = 1.0E-8, 
                           double rtol = 1.0E-6, unsigned int maxit = 100) {
  const int n = A.n;
  const int m = A.m;
  
  assert(n == m && m == b.size() && n == x.size() && "Dimensions mismatch!");
  
  Eigen::VectorXd x_new = x;
  
  for(int i = 0; i < maxit; ++i) {
    bool wellDefined = GaussSeidelstep_crs(A, b, x_new); //x_new contains next estimate
    
    if(!wellDefined) { return false; } //Check if the iteration was well-defined
    
    double deltaNorm = (x_new - x).norm();
    double xnewNorm = x_new.norm();
    
    if((deltaNorm <= atol) || (deltaNorm <= rtol * xnewNorm)) {
      x = x_new;
      return true;
    }
    x = x_new;
  }
  return false; //We did find a good enough solution in maxit steps
}*/
/*
Eigen::VectorXd lsqEst(const::Eigen::VectorXd& z, const Eigen::VectorXd& c) {
  
  const int n = z.size();
  
  assert(n == c.size() && "Input mismatch!");
  
  //Setup the matrix A
  Eigen::MatrixXd A(n, 2);
  A(0, 0) = z(0); 
  A(0, 1) = z(1);
  A(n - 1, 0) = z(n - 1);
  A(n - 1, 1) = z(n - 2);
  
  for(int i = 1; i < n - 1; ++i) {
    A(i, 0) = z(i);
    A(i, 1) = z(i - 1) + z(i + 1);
  }
  
  //Setup the vector x, c
  Eigen::VectorXd x(2);
  
  x = (A.transpose() * A).llt().solve(A.transpose() * c);
  
  return x;
}

Eigen::SparseMatrix<double> spai(Eigen::SparseMatrix<double>& A) {
  //Check dimensions
  assert(A.rows() == A.cols() && "Matrix is not square!");
  const int N = A.rows();
  //Set up matrix B - result matrix
  Eigen::SparseMatrix<double> B(N, N);
  //Mandatory makeCompressed() call to ensure column compressed storage format
  A.makeCompressed();
  //Get all important arrays for the CCS format - we treat this like arrays
  double* valPtr = A.valuePtr();
  int* innPtr = A.innerIndexPtr();
  int* outPtr = A.outerIndexPtr();
  //We store the entries of B first as triplets and then construct B at the end
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(A.nonZeros()); //Ensures that we have enough capacity
  //For each column in A (or X in the formula derived) we solve the LLS
  for(int i = 0; i < N; ++i) {
    //The number of elements between this and the next col is the number of nnz
    int nnz_i = outPtr[i + 1] - outPtr[i];
    //If the entire column is zero (N x N) matrix -> non regular
    if(nnz_i == 0) { continue; } //Skip LLS of this column
    //Create the specific matrix for the i-th column
    Eigen::SparseMatrix<double> A_i(N, nnz_i);
    std::vector<Eigen::Triplet<double>> A_i_triplets; //We again use triplets and then build A_i
    //We have at most n non zeros per column / row hence nnz_i * nnz_i is enough
    A_i_triplets.reserve(nnz_i * nnz_i); 
    //For each non-zero entry in A 
    for(int k = outPtr[i]; k < outPtr[i + 1]; ++k) {
      int row_k = innPtr[k]; //Corresponding row in the underlying matrix
      //We plan on removing for each zero in the column of A a corresponding row in A 
      int nnz_k = outPtr[row_k + 1] - outPtr[row_k]; 
      
      for(int l = 0; l < nnz_k; ++l) {
        int innIdx = outPtr[row_k] + l;
        A_i_triplets.emplace_back(
                                 //column            row              value
          Eigen::Triplet<double>(innPtr[innIdx], k - outPtr[i], valPtr[innIdx]));
      }
    }
    A_i.setFromTriplets(A_i_triplets.begin(), A_i_triplets.end());
    A_i.makeCompressed(); //Ensure again CCS format
    //Use the built-in specialized LU solver for sparse matrices
    Eigen::SparseLU<Eigen::SparseMatrix<double>> LU(A_i.transpose() * A_i);
    Eigen::VectorXd b_i = LU.solve(A_i.row(i).transpose());
    
    //Place solution as triplets into B (currently stored in triplets)
    for(int k = 0; k < b_i.size(); ++k) {
      // outPtr[i] base of curr col in innPtr  column              row value 
      triplets.emplace_back(Eigen::Triplet<double>(innPtr[outPtr[i] + k], i, b_i(k)));
    }
  }
  //Build B from the triplets
  B.setFromTriplets(triplets.begin(), triplets.end());
  B.makeCompressed();
  
  return B;
}

Eigen::VectorXd polyMult_naive_sol(const Eigen::VectorXd &u,
                               const Eigen::VectorXd &v) {
  // Fetch degrees of input polynomials
  const int degu = u.size() - 1;
  const int degv = v.size() - 1;

  // Object for product polynomial p = uv
  const int degp = degu + degv;

  Eigen::VectorXd uv(degp + 1);

  // TODO: (4-3.a) Multiply polynomials $u$ and $v$ naively.
  // START
  for (int i = 0; i <= degp; ++i) {
    const int fst = std::max(0, i - degv);
    const int lst = std::min(degu, i);
    for (int j = fst; j <= lst; ++j) {
      uv(i) += u(j) * v(i - j);
    }
  }
  // END

  return uv;
}


Eigen::VectorXd polyMult_naive(const Eigen::VectorXd &u,
                               const Eigen::VectorXd &v) {
  // Fetch degrees of input polynomials
  const int degu = u.size() - 1; //m - 1
  const int degv = v.size() - 1; //n - 1

  // Object for product polynomial p = uv
  const int degp = degu + degv;
  Eigen::VectorXd uv = Eigen::VectorXd::Zero(degp + 1); 

  // TODO: (4-3.a) Multiply polynomials $u$ and $v$ naively.
  // START
  for(int i = 0; i <= degp; ++i) {
    int bottom = std::max(0, i - degv);
    int top = std::min(i, degu);
  
    for(int a = bottom; a <= top; ++a) {
      uv(i) += u(a) * v(i - a);
    }
  }
  // END

  return uv;
}
*/
/* SAM_LISTING_BEGIN_1 */ /*
Eigen::VectorXd polyMult_fast(const Eigen::VectorXd &u,
                              const Eigen::VectorXd &v) {
  // Initialization
  const unsigned int m = u.size();
  const unsigned int n = v.size();

  Eigen::VectorXcd u_tmp(m + n - 1);
  u_tmp.head(m) = u.cast<std::complex<double>>();
  u_tmp.tail(n - 1).setZero();

  Eigen::VectorXcd v_tmp(m + n - 1);
  v_tmp.head(n) = v.cast<std::complex<double>>();
  v_tmp.tail(m - 1).setZero();

  Eigen::VectorXd uv;

  // TODO: (4-3.b) Multiply polynomials $u$ and $v$ efficiently.
  // START
  Eigen::FFT<double> fft;
  uv = fft.inv(((fft.fwd(u_tmp)).cwiseProduct(fft.fwd(v_tmp))).eval()).real();
  // END

  return uv;
}

Eigen::VectorXd polyDiv(const Eigen::VectorXd &p, const Eigen::VectorXd &u) {
  // Initialization
  const unsigned int p_size = p.size();
  const unsigned int u_size = u.size();
  // need well behaved input
  if (p_size < u_size) {
    std::cerr << "uv can't be divided by u\n";
    return Eigen::VectorXd(0);
  }

  const unsigned int v_size = p_size - u_size + 1; //degree of result

  Eigen::VectorXd v;

  //Componentwise division requires same length vectors
  Eigen::VectorXcd p_tmp = Eigen::VectorXcd::Zero(p_size);
  p_tmp.head(p_size) = p.cast<std::complex<double>>();
  
  Eigen::VectorXcd u_tmp = Eigen::VectorXcd::Zero(p_size);
  u_tmp.head(u_size) = u.cast<std::complex<double>>();
  
  Eigen::FFT<double> fft;
  Eigen::VectorXcd p_fft = fft.fwd(p_tmp);
  Eigen::VectorXcd u_fft = fft.fwd(u_tmp);
  
  //Check divisibility 1st problem -> Check the input
  for(int i = 0; i < p_size; ++i) {
    if(std::abs(u_fft(i)) < 1E-13) {
      if(std::abs(p_fft(i)) < 1E-13) {
        //We can fix the problem by not having division by zero
        u_fft(i) = 1.0;
        p_fft(i) = 0.0;
      } else {
        std::cerr << "uv can't be divided by u\n";
        return Eigen::VectorXd(0);
      }
    }
  }
  //Do the computation eval() makes sure that cwiseQuotient does not do lazy evaluation
  v = fft.inv(((p_fft.cwiseQuotient(u_fft)).eval())).real();
  
  //Check divisibility 2nd problem -> Check the result
  for(int i = v_size; i < p_size; ++i) {
    if(std::abs(v(i)) > 1E-13) { 
      std::cerr << "uv can't be divided by u\n";
      return Eigen::VectorXd(0);
    }
  }

  return v.head(v_size);
} */
/* SAM_LISTING_END_2 */
/*
Eigen::VectorXd polyDiv_sol(const Eigen::VectorXd &uv, const Eigen::VectorXd &u) {
  // Initialization
  const unsigned int mn = uv.size();
  const unsigned int m = u.size();
  // need well behaved input
  if (mn < m) {
    std::cerr << "uv can't be divided by u\n";
    return Eigen::VectorXd(0);
  }
  const unsigned int dim = mn;
  const unsigned int n = mn - m + 1;

  Eigen::VectorXd v;

  // TODO: (4-3.e) Divide polynomials $uv$ and $u$ efficiently.
  // START

  // zero padding
  Eigen::VectorXd uv_tmp = uv;
  uv_tmp.conservativeResizeLike(Eigen::VectorXd::Zero(dim));
  Eigen::VectorXd u_tmp = u;
  u_tmp.conservativeResizeLike(Eigen::VectorXd::Zero(dim));

  Eigen::FFT<double> fft;
  Eigen::VectorXcd uv_tmp_ = fft.fwd(uv_tmp);
  Eigen::VectorXcd u_tmp_ = fft.fwd(u_tmp);

  // check divisibility: case(i)
  for (unsigned int i = 0; i < dim; ++i) {
    if (abs(uv_tmp_(i)) < 1e-13) {
      if (abs(u_tmp_(i)) < 1e-13) {
        // make cwiseQuotient at i-th position equal 0
        uv_tmp_(i) = 0.0;  // complex assignment (0., 0.)
        u_tmp_(i) = 1.0;   // complex assignment (1., 0.)
      } else {
        std::cerr << "uv can't be divided by u\n";
        return Eigen::VectorXd(0);
      }
    }
  }

  Eigen::VectorXcd tmp = uv_tmp_.cwiseQuotient(u_tmp_);

  v = fft.inv(tmp).real();
  // check divisibility: case(ii)
  for (unsigned int i = n; i < dim; ++i) {
    if (abs(v(i)) > 1e-13) {
      std::cerr << "uv can't be divided by u\n";
      return Eigen::VectorXd(0);
    }
  }

  // reshape v to a suitable size
  v.conservativeResizeLike(Eigen::VectorXd::Zero(n));
  // (mn-1) - (m-1) + 1
  // END

  return v;
}

*/
/*
void PolarDecomposition::initialize(const Eigen::MatrixXd& X) {
  assert(X.rows() >= X.cols() && "Dimension mismatch!");
  //Compute the SVD here ThinV or FullV do not matter as V is square anyway.
  Eigen::JacobiSVD<Eigen::MatrixXd> SVD(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
  
  Eigen::MatrixXd U = SVD.matrixU();
  Eigen::MatrixXd V = SVD.matrixV();
  Eigen::MatrixXd Sigma = SVD.singularValues().asDiagonal();
  
  Q_ = U * V.transpose();
  M_ = V * Sigma * V.transpose();
}
*/
/*
void PolarDecomposition_sol(const Eigen::MatrixXd &A,
                                       const Eigen::MatrixXd &B) {
  const long m = A.rows();  // No. of rows of $\cob{\VX}$
  const long n = B.rows();  // No. of columns of $\cob{\VX}$
  const long k = A.cols();  // Maximal rank of $\cob{\VX}$
  // We assume $\cob{k \leq n \leq m}$
  assert(m > n);
  assert(k <= n);
  assert(B.cols() == k);
  // TODO: (3-12.d) Implement a method to initialize the data members Q_ and M_
  // for X := AB^T = QM, with optimal complexity
  // START
  // Compute QR-decompositions in an encoded way, see \lref{par:ecovsfull}
  Eigen::HouseholderQR<Eigen::MatrixXd> QRA(A);  // cost = $\cob{O(mk^2)}$
  Eigen::HouseholderQR<Eigen::MatrixXd> QRB(B);  // cost = $\cob{O(nk^2)}$
  const Eigen::MatrixXd RA{
      QRA.matrixQR().block(0, 0, k, k).template triangularView<Eigen::Upper>()};
  const Eigen::MatrixXd RB{
      QRB.matrixQR().block(0, 0, k, k).template triangularView<Eigen::Upper>()};
  // SVD of small $\cob{k\times k}$-matrix $\cob{\VR_A\VR_B^{\top}}$
  // cost = $\cob{O(k^3)}$
  Eigen::JacobiSVD<Eigen::MatrixXd> svdh(
      RA * RB.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  // Extract kxk orthogonal factor matrices
  const Eigen::MatrixXd &Vh{svdh.matrixV()};
  const Eigen::MatrixXd &Uh{svdh.matrixU()};
  // Build auxiliary mxn block matrix
  Eigen::MatrixXd W{Eigen::MatrixXd::Zero(m, n)};  // cost = $\cob{O(mn)}$
  W.block(0, 0, k, k) = Uh * Vh.transpose();       // cost = $\cob{O(k^3)}$
  W.block(k, k, n - k, n - k) = Eigen::MatrixXd::Identity(n - k, n - k);
  // Compute the Q-factor, cost = $\cob{O(kmn+kn^2)}$
  Eigen::MatrixXd Q =
      (QRB.householderQ() * ((QRA.householderQ() * W).transpose())).transpose();
  // Small kxk matrix containing singular values on diagonal
  const auto Sigma{svdh.singularValues().asDiagonal()};
  // Form M-factor of polar decomposition
  // Auxiliary matrices
  Eigen::MatrixXd S{Eigen::MatrixXd::Zero(n, k)};
  S.block(0, 0, k, k) = Vh * Sigma * Vh.transpose();  // cost = $\cob{O(k^3)}$
  Eigen::MatrixXd T{Eigen::MatrixXd::Zero(n, n)};     // cost = $\cob{O(n^2)}$
  T.block(0, 0, k, n) =
      (QRB.householderQ() * S).transpose();   // cost = $\cob{O(kn^2)}$
  Eigen::MatrixXd M = (QRB.householderQ() * T).transpose();  // cost = $\cob{O(kn^2)}$
  // END
  
  std::cout << M << std::endl;
  std::cout << " " << std::endl;
  std::cout << Q << std::endl;
}


PolarDecomposition::PolarDecomposition(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  const int m = A.rows();
  const int n = B.rows();
  const int k = A.cols();
  
  assert(m > n && k <= n && B.cols() == k && "Dimension mismatch!");
  
  //Compute the QR decompositions
  Eigen::HouseholderQR<Eigen::MatrixXd> QR_A(A); //O(mk^2)
  Eigen::HouseholderQR<Eigen::MatrixXd> QR_B(B); //O(nk^2)
  
  //Get the upper triangular R (k x k) matrices - triang view allows for optimizations
  Eigen::MatrixXd R_A = QR_A.matrixQR().block(0, 0, k, k).triangularView<Eigen::Upper>();
  Eigen::MatrixXd R_B = QR_B.matrixQR().block(0, 0, k, k).triangularView<Eigen::Upper>();
  
  //Take the SVD of k x k matrix R_AR_B
  Eigen::JacobiSVD<Eigen::MatrixXd> svdRR(
    R_A * R_B.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  
  //Get the needed components 
  Eigen::MatrixXd U_til = svdRR.matrixU();
  Eigen::MatrixXd V_til = svdRR.matrixV();
  Eigen::MatrixXd Sigma = svdRR.singularValues().asDiagonal();
  
  //Build block matrix for Q computation
  Eigen::MatrixXd Q_block = Eigen::MatrixXd::Zero(m, n);
  Q_block.block(0, 0, k, k) = U_til * V_til.transpose();
  Q_block.block(k, k, n - k, n - k) = Eigen::MatrixXd::Identity(n - k, n - k);
  
  //Build block matrix for M computation
  Eigen::MatrixXd M_block = Eigen::MatrixXd::Zero(n, n);
  M_block.block(0, 0, k, k) = V_til * Sigma * V_til.transpose();
  
  //Construct Q - householderQ() returns a sequence of householder reflections - same as mult
  _Q = (QR_B.householderQ() * (QR_A.householderQ() * Q_block).transpose()).transpose();
  
  //Construct M - could be more optimized by ensuring that null multiplications are not made, i.e block mult
  _M = (QR_B.householderQ() * (QR_B.householderQ() * M_block).transpose()).transpose();
}
*/
/*
double newton_arctan(double x0_ = 2.0) {
  
  double limit = std::numeric_limits<double>::epsilon();
  double x_n = x0_;
  double x_new;
  double update_size = 1.0;
  
  auto h_derivative = [](double x) -> double {
    double x_pow = x * x;
    return -(-1 + x_pow + 2 * (x + x_pow * x) * std::atan(x)) / (1 + x_pow);
  };
  
  auto h = [](double x) -> double {
    return std::atan(x);
  };
  
  while(update_size > limit) {
    x_new = x_n - (h(x_n) / h_derivative(x_n));
    update_size = std::abs((x_new - x_n) / x_new);
    x_n = x_new;
  }
  
  return x_n;
}

double newton_arctan2(double x0_ = 2.0) {
  double limit = std::numeric_limits<double>::epsilon();
  double x_n = x0_;
  double x_new;
  double update_size = 1.0;
  
  auto h_derivative = [](double x) -> double {
    return 1 - 2 * x * std::atan(x);
  };
  
  auto h = [](double x) -> double {
    return 2 * x - (1 + x * x) * std::atan(x);
  };
  
  while(update_size > limit) {
    x_new = x_n - (h(x_n) / h_derivative(x_n));
    update_size = std::abs((x_new - x_n) / x_new);
    x_n = x_new;
  }
  
  return x_n;

}

double newton_arctan3(double x0_ = 2.0) {
  double x0 = x0_;
  double upd = 1;
  double eps = std::numeric_limits<double>::epsilon();
  
  while(upd > eps) {
    double x1 = 
      (-x0 + (1 - x0 * x0) * std::atan(x0)) / (1 - 2 * x0 * std::atan(x0));
    upd = std::abs((x1 - x0) / x1);
    x0 = x1;
  }
  return x0;
}

template <class Function>
double steffensen(Function &&f, double x0) {
  
  //Functor for g(x) - makes call to f must hence capture it
  auto g = [&f](double x, double f_x) -> double {
    return (f(x + f_x) - f_x) / f(x);
  };
  
  double eps = std::numeric_limits<double>::epsilon();
  double error_size = 1.0;
  double x_n = x0;
  double x_new = x0;
  
  while(error_size > eps) {
    double f_x = f(x_n);
    
    if(f_x != 0.0) {
      x_new = x_n - f_x / g(x_n, f_x);
      error_size = std::abs((x_n - x_new) / x_new);
      x_n = x_new; 
    } else {
      error_size = 0.0;
    }
  }
  
  return x_n;
}

void testSteffensen() {
  auto f = [](double x) -> double {
    return x * std::exp(x) - 1;
  };
  
  std::cout << "The function converges to: " 
            << steffensen(f, 1) << std::endl;
}

template <class Function, typename LOGGER>
double steffensen_log(Function &&f, double x0, LOGGER&& log = 
  [](double) -> void{}) {
  
  //Functor for g(x) - makes call to f must hence capture it
  auto g = [&f](double x, double f_x) -> double {
    return (f(x + f_x) - f_x) / f(x);
  };
  
  double eps = std::numeric_limits<double>::epsilon();
  double error_size = 1.0;
  double x_n = x0;
  double x_new = x0;
  
  while(error_size > eps) {
    double f_x = f(x_n);
    log(x_n); //record approximation
    
    if(f_x != 0.0) {
      x_new = x_n - f_x / g(x_n, f_x);
      error_size = std::abs((x_n - x_new) / x_new);
      x_n = x_new; 
    } else {
      error_size = 0.0;
    }
  }

  log(x_n);
  return x_n;
}

template<class Function, typename LOGGER>
double steffensen_log_sol(Function&& f, double x0, 
  LOGGER&& log = [](double) -> void {}) {
  
  double x = x0;
  log(x);
  double upd = 1;
  const double eps = std::numeric_limits<double>::epsilon();
  
  while(std::abs(upd) > eps * x) {
    const double fx = f(x);
    
    if(fx != 0) {
      upd = fx * fx / (f(x + fx) - fx);
      x -= upd;
      log(x);
    } else {
      upd = 0;
    }
  }
  return x;
}

void orderSteffensen(void) {
  std::vector<double> values;
  //Logger function
  auto log = [&values](double x) -> void {
    values.push_back(x);
  };
  //Functor to test with
  auto f = [](double x) -> double {
    return x * std::exp(x) - 1;
  };
  //Get values for x^k
  steffensen_logl(f, 2.0, log);
  //x^* exactly
  double x_star = 0.5671432904097839;
  //functor for epsilon
  auto eps = [x_star](double x) -> double {
    return std::abs(x - x_star);
  };
  //functor for p approximation
  auto approx_p = [](std::vector<double> eps, int i) -> double {
    return (std::log(eps.at(i)) - std::log(eps.at(i - 1))) / 
           (std::log(eps.at(i - 1)) - std::log(i - 2));
  };
  std::vector<double> epsilon;
  //Compute the error value
  for(int i = 0; i < values.size(); ++i) {
    if(i >= 2) {
      epsilon.push_back(eps(values.at(i)));
      double p_approx = approx_p(epsilon, i);
      
      std::cout << values.at(i) << " " << epsilon.at(i) 
                << " " << p_approx << std::endl;
    } else {
      epsilon.push_back(eps(values.at(i)));
    }
  }
}
*/
/*

void circuit(const double alpha, const double beta, 
  const Eigen::VectorXd &Uin, Eigen::VectorXd& Uout,
  double rtol = 10E-6) {
    
  const double U_T = 0.5;  
  
  //Functor for f
  auto F = [Uin, alpha, beta, U_T](Eigen::VectorXd U, int i) -> Eigen::Vector3d {
    Eigen::VectorXd f(3);
    f(0) = 3 * U(0) - U(1) - U(2);
    f(1) = 3 * U(1) - U(0) - U(2) - Uin(i);
    f(2) = 3 * U(2) - U(0) - U(1) + 
      alpha * (std::exp(beta * (U(2) - Uin(i)) / U_T) - 1);
    return f;
  };
  
  //Functor for Jacobi of F
  auto DF = [Uin, alpha, beta, U_T](Eigen::VectorXd U, int i) -> Eigen::Matrix3d {
    Eigen::MatrixXd df(3, 3);
    df << 3, -1, -1,
          -1, 3, -1,
          -1, -1, 0;
    df(2, 2) = 3 + (alpha * beta / U_T) * std::exp(beta * (U(2) - Uin(i)) / U_T);
    return df;
  };
  
  Eigen::VectorXd x(3);
  Eigen::VectorXd f(3);
  Eigen::MatrixXd df(3, 3); //Could be optimized by only setting pos 2 x 2
  
  //For all given Uin values we do this
  for(int i = 0; i < Uin.size(); ++i) {
    Eigen::VectorXd u = Eigen::VectorXd::Random(3); //Random inital value
    do {
      f = F(u, i);
      df = DF(u, i);
      x = df.fullPivLu().solve(f);
      u = u - x;
    } while(x.norm() > rtol * u.norm());
    //Put result into Uout
    Uout(i) = u(0);
  }
}

void circuit_sol(const double& alpha, const double& beta,
             const Eigen::VectorXd& Uin, Eigen::VectorXd& Uout) {
  constexpr double Ut = 0.5;
  constexpr double tau = 1e-6;
  const unsigned int n = Uin.size(); std::cout << "\n\n" << std::endl;

  // TODO: (8-7.b) Compute the output voltages for the given circuit
  // START
  double Uin_;  // Uin of current node
  // lambda function for evaluation of F
  auto F = [alpha, beta](const Eigen::VectorXd& U, const double Uin_) {
    Eigen::VectorXd f(3);
    f << 3. * U(0) - U(1) - U(2), 3. * U(1) - U(0) - U(2) - Uin_,
        3. * U(2) - U(0) - U(1) +
            alpha * (std::exp(beta * (U(2) - Uin_) / Ut) - 1);
    return f;
  };

  Eigen::MatrixXd J(3, 3);               // the Jacobian
  J << 3, -1, -1, -1, 3, -1, -1, -1, 0;  // dummy in $J(2, 2)$
  Eigen::VectorXd f(3);                  // the function

  for (unsigned int i = 0; i < n; ++i) {
    Uin_ = Uin(i);
    Eigen::VectorXd U = Eigen::VectorXd::Random(3);  // random initial guess
    Eigen::VectorXd h = Eigen::VectorXd::Ones(3);
    while (h.cwiseAbs().maxCoeff() > tau * U.norm()) {
      J(2, 2) = 3. + (alpha * beta) / Ut * std::exp(beta * (U(2) - Uin_) / Ut);
      f = F(U, Uin_);
      h = J.partialPivLu().solve(f);
      U -= h;
    }
    Uout(i) = U(0);
  }
  // END
}
*/
/*
template<typename F, typename DF>
std::vector<double> GaussNewton(Eigen::Vector4d& x, F&& f, DF&& df,
  const double tol = 1E-14) {
  
  //Store the normed values of s as they will be our result
  std::vector<double> gn_update;
  
  //Posteriori termination based on relative tolerance
  do {
    Eigen::Vector4d s = df.colPivHouseholderQr().solve(f(x));
    x -= s;
    gn_update.push_back(s.lpNorm<Eigen::Infinity>());
  } while(gn_update.back() > tol);
   
  return gn_update;
}

std::vector<Eigen::Vector2d> closedPolygonalInterpolant(std::vector<Eigen::Vector2d> &Sigma,
                                                        const Eigen::VectorXd& x) {
  const int M = x.size();
  const int n = Sigma.size();
  //x is sorted hence check the boundaries if 0 <= x_i <= 1
  assert(x(0) >= 0 && x(M - 1) < 1 && "x_i not between 0(incl) and 1(excl)!");
  
  //Define storage structures for d, delta, lambda
  std::vector<Eigen::Vector2d> d(n);
  Eigen::VectorXd delta(n);
  Eigen::VectorXd lambda(n + 1); //We have n + 1 values for lambda
  
  //Add Sigma_n = Sigma_0
  Sigma.push_back(Sigma[0]); //Size is now n + 1
  //Initalize lamda_0 = 0
  lambda[0] = 0.0;
  
  //Compute lambda values
  for(int i = 0; i < n; ++i) {
    d[i] = Sigma[i + 1] - Sigma[i];
    delta(i) = d[i].norm();
    lambda(i + 1) = lambda(i) + delta(i);
  }
  
  std::vector<Eigen::Vector2d> result(n);
  //Go over all points
  for(int i = 1, j = 0; i <= n && j < x.size(); ++i) {
    
    double x_jlambda = x(j) * lambda(n);
    //Go over all points to be interpolated using linear interpolation
    while(j < x.size() && x_jlambda <= lambda[i]) {
      Eigen::Vector2d point = (lambda(i) - x_jlambda) * Sigma[i - 1] - 
                              (x_jlambda - lambda(i - 1)) * Sigma[i];
      point = point * (1 / delta(i - 1));
      result[j] = point;
      
      ++j; //Go to next x_j
    }
  }
  Sigma.pop_back();
  return result;
}
*/
/*
std::vector<Eigen::Vector2d> closedHermiteInterpolant(std::vector<Eigen::Vector2d>& Sigma,
                                                      const Eigen::VectorXd& x) {
                                                        
  std::vector<Eigen::Vector2d> slopes(n + 1);
  
  //Compute the first slope that uses delta_n and delta_0
  slopes[0] = ((delta[n - 1] * d[0]) / delta[0] + (delta[0] * d[n - 1]) / delta[n - 1]) 
               / (delta[n - 1] + delta[0]);
  slopes[0] = slopes[0] / slopes[0].norm();
  for(int i = 1; i < n; ++i) {
    slopes[i] = ((delta[i] * d[i - 1]) / delta[i - 1] + (delta[i - 1] * d[i]) / delta[i]) 
                  / (delta[i] + delta[i - 1]);
                  
    slopes[i] = slopes[i] / slopes[i].norm();
  }
  slopes[n] = slopes[0];      
  
  std::vector<Eigen::Vector2d> result(n);
  
  //Go over all points
  for(int i = 1, j = 0; i <= n && j < x.size(); ++i) {
    
    double x_jlambda = x(j) * lambda(n);
    //Go over all points to be interpolated using linear interpolation
    while(j < x.size() && x_jlambda <= lambda[i]) {
      double coordx =
        hermloceval(x_jlambda, lambda[i - 1], Sigma[i - 1](0),
                    Sigma[i](0), slopes[i - 1](0), slopes[i](0));
      double coordy = 
        hermloceval(x_jlambda, lambda[i - 1], Sigma[i - 1](1),
                    Sigma[i](1), slopes[i - 1](1), slopes[i](1));
      res[j] = {coordx, coordy};
      ++j; //Go to next x_j
    }
  }
  
}
*/
/*
template<typename CurveFunctor>
std::pair<std::vector<Eigen::Vector2d>, 
          std::vector<Eigen::Vector2d>> adaptedHermiteInterpolant(
            Curve Functor&& c,
            unsigned int nmin,
            const Eigen::VectorXd& x, double tol = 1.0E-3) {
              
  assert(x(0) >= 0 && x(M - 1) < 1 && "x_i not between 0(incl) and 1(excl)!");              
  std::vector<Eigen::Vector2d> eval;
  std::vector<Eigen::Vector2d> Sigma;
  
  //M in the pseudocode
  Eigen::Vector t = Eigen::VectorXd::LinSpaced(nmin + 1, 0, 1);
  
  //Deviations sigma_k in the pseudocode
  Eigen::Vector dev;
  do {
    unsigned int n = t.size() - 1;
    dev.resize(n);
    Sigma.resize(n + 1);
    
    //Evaluate Sigma for the lin-spaced points in M
    for(int i = 0; i <= n; ++i) {
      Sigma[i] = c(t(i));
    }
    //Compute lambda (we omit comp of delta and d and do the comp of lambda directly)
    Eigen::VectorXd lambda(n + 1);
    Eigen::VectorXd midpt(n);
    
    lambda(0) = 0.0;
    for(int i = 0; i < n; ++i) {      //this is the norm of d_i
      lambda(i + 1) = lambda(i) + (Sigma[i + 1] - Sigma[i]).norm();
      midpt(i) = (lamda(i + 1) + lambda(i)) * 0.5;
    }
    
    //Compute the linear and cubic interpolant
    std::vector<Eigen::Vector2d> v = 
      closedPolygonalInterpolant(Sigma, midpt / lambda[n]);
    std::vector<Eigen::Vector2d> w = 
      closedHermiteInterpolant(Sigma, midpt / lambda[n]);
    //Compute the deviations
    for(int i = 0; i < n; ++i) {
      dev(i) = (v[i] - w[i]).norm();
    }
    //Computes 1 / n sum(sigma_k)
    double alpha = dev.mean();
    
    std::vector<double> t_temp;
    int num = 0; 
    for(int j = 0; j < n; ++j) {
      if(dev(j) > 0.9 * alpha) {
        t_temp.push_back((t(j) + t(j + 1)) * 0.5);
        num++;
      }
    }
    t.conservativeResize(n + 1, t_temp.size());
    
    t.tail(t_temp.size()) = Eigen::VectorXd::Map(t_temp.data(), t_temp.size());
    std::sort(t.data(), t.data() + t.size());
    
  } while(dev.maxCoeff() > tol);
  
  eval = closedHermiteInterpolant(Sigma, x);
  return std::make_pair(eval, Sigma);
}
*/

/*
Eigen::SparseMatrix<double> initA(int n) {
  //Vector of triplets
  std::vector<Eigen::Triplet<double>> triplets;
  
  int m = 0.5 * n * (n - 1);
  
  int row = 0; 
  
  int i = 0;
  
  //We construct the matrix using the block struture it has
  while(row < m) {
    for(int j = i + 1; j < n - 1; ++j) { // 1 <= i < j < n - 1
      triplets.push_back(Eigen::Triplet<double>(row, i, -1)); 
      triplets.push_back(Eigen::Triplet<double>(row, j, 1));
      ++row; //Continue with next row with two elements
    }
    
    if(row < m) {
      //Add the last element
      triplets.push_back(Eigen::Triplet<double>(row, i, -1));
      
      ++row; //Update row as well as we added a row
      i = i + 1; //Next block
    }
  }
  
  Eigen::SparseMatrix<double> sparseMatrix(m, n - 1);
  sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

  return sparseMatrix;
}


void outputSquareMatrix(Eigen::MatrixXd& matrix) {
  assert(matrix.rows() == matrix.cols() && "Matrix must be square!");
  
  int n = matrix.rows();
  
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      double element = matrix(i, j);
      
      if(element >= 0) {
        std::cout << "  " << element << " ";
      } else {
        std::cout << " " << element << " ";
      }
    }
    std::cout << std::endl;
  }
}

Eigen::SparseMatrix<double> getA_ext(int n) {
  //Vector of triplets
  std::vector<Eigen::Triplet<double>> triplets; 

  int m = 0.5 * n * (n - 1);
  
  int row = 0; 
  
  int i = 0;
  
  //We construct the matrix using the block struture it has
  while(row < m) {
    for(int j = i + 1; j < n - 1; ++j) { // 1 <= i < j < n - 1 
      //A block
      triplets.push_back(Eigen::Triplet<double>(row, i + m, -1));
      triplets.push_back(Eigen::Triplet<double>(row, j + m, 1)); 
      //A^T block
      triplets.push_back(Eigen::Triplet<double>(i + m, row, -1)); 
      triplets.push_back(Eigen::Triplet<double>(j + m, row, 1)); 
      
      ++row; //Continue with next row with two elements
    }
    
    if(row < m) {
      //Add the last element - A
      triplets.push_back(Eigen::Triplet<double>(row, i + m, -1)); 
      
      //Add the last element - A^T
      triplets.push_back(Eigen::Triplet<double>(i + m, row, -1)); 
      
      ++row; //Update row as well as we added a row
      i = i + 1; //Next block
    }
  }
  
  // -I block
  for(int i = 0; i < m; ++i) {
    triplets.push_back(Eigen::Triplet<double>(i, i, -1));
  }
  
  Eigen::SparseMatrix<double> sparseMatrix(m + n - 1, m + n - 1); 
  sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

  return sparseMatrix;
}

Eigen::VectorXd constructRighthandside(const Eigen::MatrixXd& D) {
  
  int row = 0, i = 0;
  int n = D.rows();
  int m = 0.5 * n * (n - 1);
  
  Eigen::VectorXd b(m + n - 1);
  
  while(row < m) {
    for(int j = i + 1; j < n - 1; ++j) { // 1 <= i < j < n - 1
      b(row) = D(i, j);
      ++row; //Continue with next row with two elements
    }
    
    if(row < m) {
      //Add the last element
      b(row) = D(i, n - 1);
      
      ++row; //Update row as well as we added a row
      i = i + 1; //Next block
    }
  }
  
  return b;
}

Eigen::VectorXd solveExtendedNormalEquations(const Eigen::MatrixXd& D) {
  //Assert that D is a square matrix
  assert(D.rows() == D.cols() && "Dimension mismatch!");
  
  int n = D.rows();
  
  //Get the extended block matrix A_ext
  Eigen::SparseMatrix<double> A_ext = getA_ext(n);
  
  //Construct the vector [c, 0]^T
  Eigen::VectorXd c_0 = constructRighthandside(D);
  
  //Get the LU of A_ext^T * A_ext 
  Eigen::SparseLU<Eigen::SparseMatrix<double>> LU(A_ext);
  
  //Check if we were able to solve using sparse solver - probably not needed here
  if(LU.info() != Eigen::Success) {
    throw "Matrix facorization failed";
  }
  
  //Solve the systemee
  Eigen::VectorXd b = LU.solve(c_0);
  
  //Only the x vector is the result - return the last n - 1 elements
  return b.tail(n - 1);
}

Eigen::VectorXd solveNormalEquations(const Eigen::MatrixXd& D) {
  //Make sure dimensions match
  assert(D.rows() == D.cols() && "Dimension mismatch!");
  
  int n = D.rows();
  int m = 0.5 * n * (n - 1);
  
  //Computation of (A^T * A)^-1
  Eigen::MatrixXd ATA_inv = (Eigen::MatrixXd::Identity(n - 1, n - 1) +
                             Eigen::MatrixXd::Ones(n - 1, n - 1)) * (1.0 / n); 
  
  //Computation of A^T * b using the structure of D     
  Eigen::MatrixXd D_modified = Eigen::MatrixXd::Zero(n, n);
  
  //Fill strict upper triangular part
  D_modified.triangularView<Eigen::Upper>() = 
    D.triangularView<Eigen::Upper>();
    
  //Fill strict lower triangular part
  Eigen::MatrixXd D_trans_temp = D_modified.transpose();
  D_modified = D_modified - D_trans_temp;
  
  //Get transposed of the special matrix D_modified (D in the solution)
  Eigen::MatrixXd DT = D_modified.transpose(); 
  
  //Compute A^T * b using D^T_(1:n-1, :) * 1 
  Eigen::MatrixXd ATb = DT.block(0, 0, n - 1, n) * Eigen::VectorXd::Ones(n); 
  
  //Compute and return x
  Eigen::VectorXd x = ATA_inv * ATb; 
  return x;
}

bool testNormalEquations(const Eigen::MatrixXd& D) {
  
  assert(D.rows() == D.cols() && "Dimension mismatch!");
  
  Eigen::VectorXd x_normal = solveExtendedNormalEquations(D);
  Eigen::VectorXd x_special = solveNormalEquations(D);
  
  if((x_normal - x_special).norm() < 1E-8 * x_normal.norm()) {
    return true;
  }
  return false;
}
*/
/*
Eigen::VectorXd fruitPrice() {
  //Initialize coefficent matrix
  Eigen::MatrixXd A(6, 6);
  
  A << 3,  1,  7,  2,  0,  0,
       2,  0,  0,  0,  0,  1,
       1,  0,  0,  0,  3,  0,
       0,  5,  0,  1,  0,  0,
       0,  0,  0,  2,  0,  1,
       0,  1, 20,  0,  0,  0;
       
  //Initialize right-hand side vector b
  Eigen::VectorXd b(6);
  
  b << 11.10, 17.00, 6.10, 5.25, 12.50, 7.00;
  
  //Initialize the solver - we use LU decomposition (A is square)
  Eigen::PartialPivLU<Eigen::MatrixXd> LU(A);
  
  //Compute and return the result
  Eigen::VectorXd x = LU.solve(b);
  return x;
}

Eigen::VectorXd multA(const Eigen::VectorXd& d1, const Eigen::VectorXd& d2,
                      const Eigen::VectorXd& c, const Eigen::VectorXd& x) {
  const int n = d1.size();
  
  assert(n == d2.size() && n == c.size() && 2 * n == x.size() && "Dimensions mismatch!");
  
  Eigen::VectorXd y(2 * n);
  
  y.segment(0, n) = d1.cwiseProduct(x.segment(0, n)) +
                    c.cwiseProduct(x.segment(n, n));
  y.segment(n, n) = c.cwiseProduct(x.segment(0, n)) + 
                    d2.cwiseProduct(x.segment(n, n));
  return y;        
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> computeLU(const Eigen::VectorXd& d1, 
                                                      const Eigen::VectorXd& d2,
                                                      const Eigen::VectorXd& c) {
  const int n = d1.size();
  const int twoN = 2 * n;
  
  assert(n == d2.size() && n == c.size() && "Dimensions mismatch!");
  
  //Initialize matrices with zero matrices (important)
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(twoN, twoN);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(twoN, twoN);
  
  //Set identity matrices in L
  L.block(0, 0, n, n).diagonal() = Eigen::VectorXd::Ones(n);
  L.block(n, n, n, n).diagonal() = Eigen::VectorXd::Ones(n);
  
  //Compute D_1^-1 using cwiseInverse 1 / x for each entry x
  Eigen::VectorXd D_inv = d1.cwiseInverse();
  
  //Efficiently compute the diagonal of C * D_1^-1
  Eigen::VectorXd CD_1_inv = c.cwiseProduct(D_inv);
  
  //Set entries C * D_1^-1 in L
  L.block(n, 0, n, n).diagonal() = CD_1_inv;
  
  //Set D1 and C entries in U
  U.block(0, 0, n, n).diagonal() = d1;
  U.block(0, n, n, n).diagonal() = c;
  
  //Compute (CD_1^-1) * C efficiently for the Schur complement
  Eigen::VectorXd CD_invC = CD_1_inv.cwiseProduct(c);
  
  //We could just subtract in O(n^2) as this is the overall complexity
  //But we can also do it in in O(n) - subtract only the diagonals
  Eigen::VectorXd S_diag = d2 - CD_invC;
  
  //Set the S entry in U
  U.block(n, n, n, n).diagonal() = S_diag;
  
  //return the result
  return std::make_pair(L, U);
}

Eigen::MatrixXd computeLUStorage(const Eigen::VectorXd& d1, 
                                 const Eigen::VectorXd& d2,
                                 const Eigen::VectorXd& c) {
  const int n = d1.size();
  const int twoN = 2 * n;
  
  assert(n == d2.size() && n == c.size() && "Dimensions mismatch!");
  
  //Initialize matrices with zero matrices (important)
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(twoN, twoN);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(twoN, twoN);
  
  //Set identity matrices in L
  L.block(0, 0, n, n).diagonal() = Eigen::VectorXd::Ones(n);
  L.block(n, n, n, n).diagonal() = Eigen::VectorXd::Ones(n);
  
  //Compute D_1^-1 using cwiseInverse 1 / x for each entry x
  Eigen::VectorXd D_inv = d1.cwiseInverse();
  
  //Efficiently compute the diagonal of C * D_1^-1
  Eigen::VectorXd CD_1_inv = c.cwiseProduct(D_inv);
  
  //Set entries C * D_1^-1 in L
  L.block(n, 0, n, n).diagonal() = CD_1_inv;
  
  //Set D1 and C entries in U
  U.block(0, 0, n, n).diagonal() = d1;
  U.block(0, n, n, n).diagonal() = c;
  
  //Compute (CD_1^-1) * C efficiently for the Schur complement
  Eigen::VectorXd CD_invC = CD_1_inv.cwiseProduct(c);
  
  //We could just subtract in O(n^2) as this is the overall complexity
  //But we can also do it in in O(n) - subtract only the diagonals
  Eigen::VectorXd S_diag = d2 - CD_invC;
  
  //Set the S entry in U
  U.block(n, n, n, n).diagonal() = S_diag;
  
  //return the result
  Eigen::MatrixXd result(twoN, twoN);
  result.triangularView<Eigen::Upper>() = U.triangularView<Eigen::Upper>();
  result.triangularView<Eigen::StrictlyLower>() = L.triangularView<Eigen::StrictlyLower>();
  
  return result;
}

Eigen::VectorXd solveA_sol(const Eigen::VectorXd & d1_, const Eigen::VectorXd & d2_,
                const Eigen::VectorXd & c_, const Eigen::VectorXd & b_) {
    int n = d1_.size();
    assert(n == d2_.size()
           && n == c_.size()
           && 2*n == b_.size()
           && "Size mismatch!");

    Eigen::ArrayXd c1 = c_, c2 = c_, d1 = d1_, d2 = d2_, b = b_;

    double eps = std::numeric_limits<double>::epsilon();

    // For forward elimination + pivotisation we are
    // only required to loop the first half of the matrix
    // Loop over diagonal
    for(int k = 0; k < n; ++k) {
        // Check if need to pivot (i.e. swap two rows)
        // Luckily we only need to check two rows
        double maxk = std::max(std::abs(d1(k)), std::abs(c1(k)));
        double maxnk = std::max(std::abs(c2(k)), std::abs(d2(k)));
        if( std::abs( c1(k) ) / maxk // max relative pivot at row k
            >
            std::abs( d1(k) ) / maxnk // max relative pivot at rok k+n
            ) {
            // Matrix
            std::swap(d1(k), c2(k));
            std::swap(c1(k), d2(k));
            // R.h.s.
            std::swap(b(n), b(n+k));
        }

        // Check if matrix is almost singuloar
        double piv = d1(k);
        // Norm of the block from k,k to n-1,n-1
        double norm = std::abs(d1(k)) + std::abs(d2(k))
                    + std::abs(c1(k)) + std::abs(c2(k));
        if( piv < eps * norm ) {
            //std::cout << "Warning: matrix nearly singular!" << std::endl;
        }

        // Multiplication facot:
        double fac = c2(k) / piv;

        // Actually perform substitution
        // Bottom Right poriton changes
        d2(k) -= c1(k) * fac;
        // R.h.s
        b(n+k) -= b(k) * fac;
    }

    // Now the system has the form:
    // | d1 | c  |   |   |   |   |
    // | 0  | d2 | * | x | = | b |
    // with d1, d2, c diagonal

    // Backward substitution

    // Lower potion
    b.tail(n) /= d2;
    // Upper portion
    b.head(n) = (b.head(n) - c1*b.tail(n)) / d1;

    return b;

}


Eigen::VectorXd solveA(const Eigen::VectorXd& d1, const Eigen::VectorXd& d2,
                       const Eigen::VectorXd& c, const Eigen::VectorXd& b) {
  const int n = d1.size();
  
  assert(n == d2.size() && n == c.size() && b.size() == 2 * n && "Dimensions mismatch!");
  
  //The result vector x
  Eigen::VectorXd x(2 * n);
  
  //We compute (using the formula found) two entries of x per iteration
  for(int i = 0; i < n; ++i) {
    //Compute and check the determinant of the block
    double det = d1(i) * d2(i) - c(i) * c(i);
    //Check if almost singular where it would mess up the result
    if(std::abs(det) < std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("The matrix is not regular!");
    }
    
    //Compute 1 / det if det != 0
    det = 1 / det;
    
    //Set the entries of x without multiplying by 1 / det (do afterwards)
    x(i) = d2(i) * b(i) - c(i) * b(i + n);
    x(i + n) = -c(i) * b(i) + d1(i) * b(i + n);
    
    //Multiply with 1 / det (is stored in det)
    x(i) = det * x(i);
    x(i + n) = det * x(i + n);
  }
  //return result
  return x;
}

//Helper method is only called from strasseMatMult and dimension are checked && n = 2^k
Eigen::MatrixXd strassenMatMultRec(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  //Get n and k - we already checked that l must always exist
  const int n = A.rows();
  const int l = n / 2; //Will always be integer
  //check if base case - do normal matrix multiplication
  if(l == 1) {
    Eigen::MatrixXd C = A * B; //Ensures no lazy eval happens
    return C;
  } 
  //Compute blocks A_11, A_12, ... , B_21, B_22 (inefficient but makes code easier to understand)
  Eigen::MatrixXd A11 = A.block(0, 0, l, l);
  Eigen::MatrixXd A12 = A.block(0, l, l, l);
  Eigen::MatrixXd A21 = A.block(l, 0, l, l);
  Eigen::MatrixXd A22 = A.block(l, l, l, l);
  
  Eigen::MatrixXd B11 = B.block(0, 0, l, l);
  Eigen::MatrixXd B12 = B.block(0, l, l, l);
  Eigen::MatrixXd B21 = B.block(l, 0, l, l);
  Eigen::MatrixXd B22 = B.block(l, l, l, l);
  
  //Compute the block Q_0, ..., Q_6
  Eigen::MatrixXd Q0 = strassenMatMultRec( (A11 + A22), (B11 + B22) );
  Eigen::MatrixXd Q1 = strassenMatMultRec( (A21 + A22), B11 );
  Eigen::MatrixXd Q2 = strassenMatMultRec( A11, (B12 - B22) );
  Eigen::MatrixXd Q3 = strassenMatMultRec( A22, (-B11 + B21) );
  Eigen::MatrixXd Q4 = strassenMatMultRec( (A11 + A12), B22 );
  Eigen::MatrixXd Q5 = strassenMatMultRec( (-A11 + A21), (B11 + B12) );
  Eigen::MatrixXd Q6 = strassenMatMultRec( (A12 - A22), (B21 + B22) );
  
  //Put the blocks together into the result
  Eigen::MatrixXd C(n, n);
  
  C.block(0, 0, l, l) = Q0 + Q3 - Q4 + Q6;
  C.block(l, 0, l, l) = Q1 + Q3;
  C.block(0, l, l, l) = Q2 + Q4;
  C.block(l, l, l, l) = Q0 + Q2 - Q1 + Q5;
  
  //return the result;
  return C;
}
Eigen::MatrixXd strassenMatMult(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  const int n = A.rows();
  //Make sure dimensions match
  assert(n == A.cols() && n == B.rows() && n == B.cols() && "Dimension mismatch!");
  //We check if n is a power of two - use log_2(x) = log(x) / log(2)
  double k_check = std::log(n) / std::log(2.0);
  //Check if floor is equal to ceil -> is integer, must also be positive because otherwise fraction
  if(std::floor(k_check) != std::floor(k_check)) {
    throw std::runtime_error("n is not power of 2");
  }
  //return the result computed by the recursive helper method
  return strassenMatMultRec(A, B);
}

void strassenImplCorrectness(double tol = 1E-8) {
  //Check between 2^1 and 2^9
  for(int i = 2; i = i * 2; i < (1 << 10)) {
    //Check for 10 random matrices
    
    for(int j = 0; j < 10; ++j) {
      std::cout << "Test: size = " << i << " current iteration of test: " << j << std::endl;
      Eigen::MatrixXd A = Eigen::MatrixXd::Random(i, i);
      Eigen::MatrixXd B = Eigen::MatrixXd::Random(i, i);
      
      Eigen::MatrixXd C1 = A * B;
      Eigen::MatrixXd C2 = strassenMatMult(A, B);
      
      if((C1 - C2).norm() > tol * C1.norm()) { //not the same result
        std::cout << "Not passed error was: " << (C1 - C2).norm() << std::endl;
      } else {
        std::cout << "Passed the test" << std::endl;
      }
    }
  }
}

Eigen::MatrixXcd matPow(Eigen::MatrixXcd& A, unsigned int k) {
  assert(A.rows() == A.cols() && "Dimensions mismatch!");
  
  //Check if base case
  if(k == 1) {
    return A;
  }

  //Check if k = 2 * l
  if(k % 2 == 0) { //k = 2 * l
    int l = k / 2;
    return matPow(A, l) * matPow(A, l);
  } else { //k = 2 * l + 1 - division rounds down
    int l = k / 2;
    return (matPow(A, l) * matPow(A, l)) * A;
  }
}

void sinhError() {
  //The numerically unstable sinh functor 
  auto sinh_unstable = [](double x) -> double {
    double t = std::exp(x);
    return 0.5 * (t - 1.0 / t);
  };
  
  for(int k = 1; k <= 10; ++k) {
    double x = 1 / std::pow(10, k); //10^-k
    
    double sinhUnstable = sinh_unstable(x);
    double sinhExact = std::sinh(x);
    
    double eps_rel = std::abs(sinhUnstable - sinhExact) / std::abs(sinhExact);
    
    std::cout << "Test for: x = 10^-" << k << " relative error = " << eps_rel << std::endl;
  }
  
}

Eigen::MatrixXd buildA(const Eigen::VectorXd& a) {
  
  const int n = a.size();
  //Initialize a zero matrix first 
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  
  //For each row fill with the until diagonal element is reached
  for(int row = 0; row < n; ++row) {
    for(int col = 0; col <= row; ++col) { //corresponds to element in a
      A(row, col) = a(col);
    }
  }
  
  return A;
}

Eigen::MatrixXd buildA_alt(const Eigen::VectorXd& a) {
  
  const int n = a.size();
  //Initialize a zero matrix first 
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  
  //For each row fill with the until diagonal element is reached
  for(int row = 0; row < n; ++row) {
    A.row(row).head(row + 1) = a.head(row + 1); //Amount = maxIndex + 1
  }
  
  return A;
}

void solveA(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
    Eigen::VectorXd& x) {
  //Check the dimensions
  assert(a.size() == b.size() && a.size() == x.size() && "Dimension mismatch!");
  //Build the matrix A
  Eigen::MatrixXd A = buildA(a);
  
  //Get the LU decomposition - use its solver method afterwards
  Eigen::PartialPivLU<Eigen::MatrixXd> LU(A);
  
  //Solve the system using the solve method
  x = LU.solve(b);
}

void solveA_fast(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
  Eigen::VectorXd& x) {
  
  const int n = a.size();
  
  assert(n == b.size() && "Dimension mismatch!"); //We set x ourselves
  
  //Initialize the matrix because we do not size check
  x = Eigen::VectorXd(n);
  //Set first element manually
  x(0) = b(0) / a(0);
  
  //Use the formula we found to compute the x_i
  for(int i = 1; i < n; ++i) {
    x(i) = b(i - 1) / a(i) + b(i) / a(i);
  }
}

void solveA_triangular(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
  Eigen::VectorXd& x) {
  //No dimension checks done - must be added for full implementation
  Eigen::MatrixXd A = buildA(a);
  x = A.triangularView<Eigen::Lower>().solve(b);
}

void CSS(const Eigen::MatrixXd& A, Eigen::VectorXd& val, 
         Eigen::VectorXd& row_ind, Eigen::VectorXd& col_ptr) {
  
  const int n = A.rows();
  
  //Assert we get square matrix
  assert(n == A.cols() && "Dimension mismatch!");
  
  int nnz = 0;
  
  //Move through array once and count the non-zero elements 
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      if(A(i, j) != 0) {
        ++nnz;
      }
    }
  }
  
  //We now know all dimensions - as in definition
  val = Eigen::VectorXd(nnz);
  row_ind = Eigen::VectorXd(nnz);
  col_ptr = Eigen::VectorXd(n + 1);
  
  int val_ptr = 0; //Same for row_ind
  int col_ptr_index = 0; //Used to navigate in col_ptr
  
  //Fill the vectors
  for(int col = 0; col < n; ++col) {
    bool first = true; 
    
    for(int row = 0; row < n; ++row) {
      if(first && A(row, col) != 0) { //First nnz of the column
        col_ptr[col_ptr_index] = row; //Store index of first nnz
        val[val_ptr] = A(row, col);
        row_ind[val_ptr] = row;
        
        ++col_ptr_index;
        ++val_ptr;
        first = false;
        
      } else if(A(row, col) != 0) { //nnz but not first
        val[val_ptr] = A(row, col);
        row_ind[val_ptr] = row;
        
        ++val_ptr;
      }
    }
  }
  
  //Set last entry in col_ptr
  col_ptr[col_ptr_index] = nnz;
}

void GSIt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
          Eigen::VectorXd& x, double rtol = 1E-8) {
  const int n = A.rows();
  
  //Assert that dimensions match
  assert(n == A.cols() && n == b.size() && n == x.size() && "Dimension mismatch!");
  
  //First get matrices D, L, U
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, n);
  
  D.diagonal() = A.diagonal();
  L.triangularView<Eigen::StrictlyLower>() = A.triangularView<Eigen::StrictlyLower>();
  U.triangularView<Eigen::StrictlyUpper>() = A.triangularView<Eigen::StrictlyUpper>();
  //Next step in the iteration - used to compare with
  Eigen::VectorXd x_new = x;
  
  //Create TriangularView objects so we can directly apply the solver
  Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Upper> UD = A.triangularView<Eigen::Upper>();
  Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower> LD = A.triangularView<Eigen::Lower>();
  
  //Use a correction based terminated do / while loop
  do {
    x = x_new; 
    x_new = UD.solve(b) - UD.solve(L * LD.solve(b - U * x));
  } while((x_new - x).norm() > rtol * x_new.norm());
}

void GSIt_sol(const Eigen::MatrixXd & A, const Eigen::VectorXd & b,
          Eigen::VectorXd & x, double rtol = 1E-8) {
    const auto U = Eigen::TriangularView<const Eigen::MatrixXd, Eigen::StrictlyUpper>(A);
    const auto L = Eigen::TriangularView<const Eigen::MatrixXd, Eigen::StrictlyLower>(A);

    const auto UpD = Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Upper>(A);
    const auto LpD = Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower>(A);

    // A temporary vector to store result of iteration
    Eigen::VectorXd temp(x.size());

    // We'll use pointer magic to
    Eigen::VectorXd* xold = &x;
    Eigen::VectorXd* xnew = &temp;

    // Iteration counter
    unsigned int k = 0;
    double err;

#if VERBOSE
        std::cout << std::setw(10) << "it."
                  << std::setw(15) << "err" << std::endl;
#endif // VERBOSE
    do {
        // Compute next iteration step
        *xnew = UpD.solve(b) - UpD.solve(L*LpD.solve(b - U * (*xold) ));

        // Absolute error
        err = (*xold - *xnew).norm();
#if VERBOSE
        std::cout << std::setw(10) << k++
                  << std::setw(15) << std::setprecision(3) << std::scientific
                  << err << std::endl;
#endif // VERBOSE

        // Swap role of previous/next iteration
        std::swap(xold, xnew);
    } while( err > rtol * (*xnew).norm() );

    x = *xnew;

    return;
}

void testImplGSIt(int n, double rtol = 1E-8) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  
  //Construct matrix - for row i we want to set (i, i - 1), (i, i), (i, i + 1) if possible
  for(int i = 0; i < n; ++i) {
    for(int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j) {
      if(i == j) {
        A(i, j) = 3;
      } else if(i < j) {
        A(i, j) = 1;
      } else if(i > j) {
        A(i, j) = 2;
      }
    }
  }
  //Initial guess is b
  Eigen::VectorXd x = b;
  //Compute approximation
  GSIt(A, b, x, rtol);
  
  //Output residual
  std::cout << "Residual of approximation: " << (b - A * x).norm() << std::endl;
}

*/
/*
template<typename Vector>
Eigen::SparseMatrix<double> solveDiagSylvesterEq(const Vector diagA) {
  
  const int n = diagA.size();
  //Compute the the values x_ii and save them in a vector first
  Eigen::VectorXd x(n);
  
  for(int i = 0; i < n; ++i) {
    //Check if a_ii = 0 -> A is not regular 
    if(std::abs(diagA(i)) < std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("A is not regular!");
    }
    
    x(i) = 1 / (diagA(i) + 1 / diagA(i));
  }
  //Initialize matrix
  Eigen::SparseMatrix<double> result(n, n);
  //Reserve n enties
  result.reserve(n);
  //Insert all diagonal entries - the nnzs
  for(int i = 0; i < n; ++i) {
    result.insert(i, i) = x(i);
  }
  //Make matrix compressed in CCF
  result.makeCompressed();
  
  return result;
}

Eigen::SparseMatrix<double> sparseKron(const Eigen::SparseMatrix<double>& M) {
  //Check that the matrix is square
  assert(M.rows() == M.cols() && "Dimension mismatch!");
  //Get the arrays, which are given as double pointer - treat as arrays later
  const double* val = M.valuePtr();
  const int* row_ind = M.innerIndexPtr();
  const int* col_ptr = M.outerIndexPtr();
  
  //Initialize SparseMatrix and reserve space 
  int nnz = M.nonZeros();
  int n = M.rows();
  int outerSize = M.outerSize(); //Size of val / row_ind
  Eigen::SparseMatrix<double> M_kronProd_M(n * n, n * n);
  M_kronProd_M.reserve(nnz * nnz);
  
  //Iterate over the sparse matrix using a nested loop + offset
  for(int i = 0; i < outerSize; ++i) {
    for(int j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
      //For each non-zero entry we have a block to fill
      int row = row_ind[j]; //Corresponding row in M
      int col = i;
      double m_ij = val[j]; //Corresponding element in M
      
      //Insert block with the necessary offset
      for(int a = 0; a < outerSize; ++a) {
        for(int b = col_ptr[a]; b < col_ptr[a + 1]; ++b) {
          //These offsets are given by m_ij and tell us the current block
          int colOffset = col * n;
          int rowOffset = row * n;
          //Then we get the current index by these indices
          int innerCol = a;
          int innerRow = row_ind[b];
          //Value we want to set in the inner block
          double value = val[b];
          //Set value accordingly
          M_kronProd_M.insert(colOffset + innerCol, rowOffset + innerRow) = m_ij * value;
        }
      }
    }
  }
  //Make SparseMatrix compressed and turn into CFF format
  M_kronProd_M.makeCompressed();
  //return result
  return M_kronProd_M;
}

Eigen::SparseMatrix<double> sparseKronNotCompressed(const Eigen::SparseMatrix<double>& M) {
  //Check that the matrix is square
  assert(M.rows() == M.cols() && "Dimension mismatch!");
  //Get the arrays, which are given as double pointer - treat as arrays later
  const double* val = M.valuePtr();
  const int* row_ind = M.innerIndexPtr();
  const int* col_ptr = M.outerIndexPtr();
  
  //Initialize SparseMatrix and reserve space 
  int nnz = M.nonZeros();
  int n = M.rows();
  int outerSize = M.outerSize(); //Size of val / row_ind
  Eigen::SparseMatrix<double> M_kronProd_M(n * n, n * n);
  M_kronProd_M.reserve(nnz * nnz + nnz);
  
  //Iterate over the sparse matrix using a nested loop + offset
  for(int i = 0; i < outerSize; ++i) {
    for(int j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
      //For each non-zero entry we have a block to fill
      int row = row_ind[j]; //Corresponding row in M
      int col = i;
      double m_ij = val[j]; //Corresponding element in M
      
      //Insert block with the necessary offset
      for(int a = 0; a < outerSize; ++a) {
        for(int b = col_ptr[a]; b < col_ptr[a + 1]; ++b) {
          //These offsets are given by m_ij and tell us the current block
          int colOffset = col * n;
          int rowOffset = row * n;
          //Then we get the current index by these indices
          int innerCol = a;
          int innerRow = row_ind[b];
          //Value we want to set in the inner block
          double value = val[b];
          //Set value accordingly
          M_kronProd_M.insert(colOffset + innerCol, rowOffset + innerRow) = m_ij * value;
        }
      }
    }
  }
  //return result
  return M_kronProd_M;
}


Eigen::MatrixXd solveSpecialSylvesterEq(const Eigen::SparseMatrix<double>& A) {
  const int n = A.rows();
  //Check if A is square
  assert(n == A.cols() && "Dimension mismatch");
  //X is the result 
  Eigen::MatrixXd X(n, n);
  //Build left-hand side - this method does not yet call makeCompressed()
  //And it also reserves the diagonal just to make sure no expensive ops are made
  Eigen::SparseMatrix<double> C = sparseKronNotCompressed(A);
  //Set the diagonal
  for(int i = 0; i < n * n; ++i) {
    C.insert(i, i) = C.coeff(i, i) + 1;
  }
  //Extract arrays from sparse matrix
  const int* col_ptr = A.outerIndexPtr();
  const int* row_ind = A.innerIndexPtr();
  const double* val = A.valuePtr();
  //Build right hand side
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n * n);
  
  for(int i = 0; i < n; ++i) {
    for(int j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
      int col = i;
      int row = row_ind[j];
      double value = val[j];
      
      b[col * n + row] = value;
    }
  }
  //We first make the matrix C compressed
  C.makeCompressed();
  //We can now apply sparse solvers to it
  Eigen::SparseLU<Eigen::SparseMatrix<double>> LU(C);
  //We solve for x which is vec(X)
  Eigen::VectorXd x = LU.solve(b);
  //We map vec x back to X
  for(int i = 0; i < n; ++i) {
    X.col(i) = x.segment(i * n, n); //segment starting at i * n of length n
  }
  return X;
}
*/
/*

class NodalPotentials {
  public:
    NodalPotentials() = delete;
    NodalPotentials(const NodalPotentials&) = default;
    NodalPotentials(NodalPotentials&&) = default;
    NodalPotentials& operator=(const NodalPotentials&) = default;
    NodalPotentials& operator=(NodalPotentials&&) = default;
    ~NodalPotentials() = default;
    
    NodalPotentials(double R, double Rx);
    Eigen::VectorXd operator() (double V) const;
    
    private:
      Eigen::MatrixXd A;
      Eigen::PartialPivLU<Eigen::MatrixXd> LU;
      Eigen::VectorXd nodalBaseVoltage;
};

NodalPotentials::NodalPotentials(double R, double Rx) {
  double zeta = R / Rx;
  
  A = Eigen::MatrixXd(15, 15);
  
  //Initialize A with the given structure
  A <<  2, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       -1,  4, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,
        0, -1,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,
        0,  0, -1,  3,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,
       -1, -1,  0,  0,  4,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,
        0,  0,  0, -1,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0, -1,
        0,  0,  0,  0, -1,  0,  4,  0,  0, -1, -1,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  4, -1,  0,  0, -1, -1,  0, -1,
        0,  0,  0,  0,  0, -1,  0, -1,  3,  0,  0,  0, -1,  0,  0,
        0,  0,  0,  0,  0,  0, -1,  0,  0,  2, -1,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  4, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  3, -1,  0,  0,
        0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1,  3,  0,  0,
        0, -1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  3 + zeta, zeta,
        0,  0, -1, -1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  -zeta , 4 + zeta;
        
  LU = A.partialPivLu(); //Store LU decomposition for future usage 
  
  //Solve the base voltage LSE and store the result
  Eigen::VectorXd b = Eigen::VectorXd::Zero(15);
  b(5) = 1; //i = 6 but 0-indexed
  nodalBaseVoltage = LU.solve(b);
}

Eigen::VectorXd NodalPotentials::operator() (double V) const {
  //We use linearity of the voltage relationship and the base
  //value to make this computation even more efficient
  Eigen::VectorXd result = nodalBaseVoltage * V;
  return result;
}

class ImpedanceMap {
  public:
    ImpedanceMap() = delete;
    ImpedanceMap(const ImpedanceMap&) = default;
    ImpedanceMap(ImpedanceMap&&) = default;
    ImpedanceMap& operator=(const ImpedanceMap&) = default;
    ImpedanceMap& operator=(ImpedanceMap&&) = default;
    ~ImpedanceMap() = default;
    
    ImpedanceMap(double R, double V);
    double operator() (double Rx) const;
    
    private:
      Eigen::MatrixXd A_0;
      Eigen::PartialPivLU<Eigen::MatrixXd> LU;
      
      Eigen::VectorXd w;
      Eigen::VectorXd z;
      double alpha;
      double beta;
      double R;
      double V;
};

ImpedanceMap::ImpedanceMap(double R, double V) {
  
  A_0 = Eigen::MatrixXd(15, 15);
  
  //Initialize A_0 with the given structure
  A_0 <<  2, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         -1,  4, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,
          0, -1,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,
          0,  0, -1,  3,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,
         -1, -1,  0,  0,  4,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,
          0,  0,  0, -1,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0, -1,
          0,  0,  0,  0, -1,  0,  4,  0,  0, -1, -1,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  4, -1,  0,  0, -1, -1,  0, -1,
          0,  0,  0,  0,  0, -1,  0, -1,  3,  0,  0,  0, -1,  0,  0,
          0,  0,  0,  0,  0,  0, -1,  0,  0,  2, -1,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  4, -1,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  3, -1,  0,  0,
          0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1,  3,  0,  0,
          0, -1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  3,  0,
          0,  0, -1, -1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0 , 4;
          
  //Initialize the LU decomposition
  LU = A_0.partialPivLu();
  //Compute u_0 
  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(15);
  u0(13) = -1;
  u0(14) = 1;
  //Compute b
  Eigen::VectorXd b = Eigen::VectorXd::Zero(15);
  b(5) = V;
  
  //z = A_0^-1 * u0 <-> A_0 * z = u0
  z = LU.solve(u0);
  //w = A_0^-1 * b <-> A_0 * w = b
  w = LU.solve(b);
  //alpha = u_0^T * z - could use .dot() here as well
  alpha = u0.transpose() * z;
  //beta = u_0^T * w - could use .dot() here as well
  beta = u0.transpose() * w;
}

double ImpedanceMap::operator() (double Rx) const {
  
  double zeta = R / Rx;
  
  //Contains all voltages U1, ... U15 after computation
  Eigen::VectorXd x = w - ((zeta * beta) / (1 + zeta * alpha)) * z;
  
  //Compute I_V
  double I_V = (x(5) - V) / R; //U(5) is 0-indexed 6th element
  
  //Compute and return impedance
  double impedance = V / I_V;
  return impedance;
}

void applyHouseholder(Eigen::VectorXd& x, const Eigen::VectorXd& v) {
  double vTx = v.transpose() * x; //v^T * x
  double vTv = v.transpose() * v; //v^T * v
  
  x = x - 2 * (v / vTv) * vTx;
}

template<typename Scalar>
void applyHouseholder(Eigen::VectorXd& x, const Eigen::MatrixBase<Scalar>& V) {
  
  const int n = x.size();
  
  for(int i = 0; i < n; ++i) {
    //First set the part from i + 1 to n
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
    v.tail(i - 1) = V.col(i).tail(i - 1);
    //Compute (v_i)_i
    v(i) = std::sqrt(1.0 - v.squaredNorm());
    
    //Apply H_{i}^{-1} = H_{i}
    applyHouseholder(x, v);
  }
}
*/



Eigen::SparseMatrix<double> buildC(const Eigen::MatrixXd& A) {
  const int n = A.rows();
  //Assert that the input is a square matrix
  assert(n == A.cols() && "Dimension mismatch!");
  //Container for triplets
  std::vector<Eigen::Triplet<double>> triplets;
  
  //Add triplets from I x A
  for(int i = 0; i < n; ++i) { //Only diagonal entries are non-zero
    //Add matrix block
    for(int row = 0; row < n; ++row) {
      for(int col = 0; col < n; ++col) { 
        int offset = i * n;
        triplets.push_back(Eigen::Triplet<double>(offset + row, offset + col, 1 * A(row, col)));
      }
    }
  }
  
  //Add triplets from A x I
  for(int row = 0; row < n; ++row) {
    for(int col = 0; col < n; ++col) {
      //Only diagonal entries of I are non-zero
      for(int i = 0; i < n; ++i) {
        int offsetRow = row * n;
        int offsetCol = col * n;
        triplets.push_back(Eigen::Triplet<double>(offsetRow + i, offsetCol + i, A(row, col) * 1));
      }
    }
  }
  
  //Build the sparse matrix from the triplets
  Eigen::SparseMatrix<double> result(n * n, n * n);
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed(); //Ensures CCF format 
  return result;
}

Eigen::VectorXd buildB(int n) {
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n * n);
  
  for(int i = 0; i < n; ++i) {
    b(i * (n + 1)) = 1;
  }
  
  return b;
}

Eigen::VectorXd buildBWithReshape(int n) {
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd b = Eigen::Map<Eigen::MatrixXd>(B.data(), n * n, 1);
  
  return b;
}

void solveLyapunov(const Eigen::MatrixXd& A, Eigen::MatrixXd& X) {
  const int n = A.rows();
  //Get C from the method
  Eigen::SparseMatrix<double> C = buildC(A);
  //Build b
  Eigen::VectorXd b = buildB(n);
  //x will contain vec(X)
  Eigen::VectorXd x;
  //LU based solver
  Eigen::SparseLU<Eigen::SparseMatrix<double>> LU(C);
  x = LU.solve(b);
  //QR based solver
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QR(C);
  x = QR.solve(b);
  
  //Reshape the vector to a matrix
  X = Eigen::Map<Eigen::MatrixXd>(x.data(), n, n);
}

bool testLyapunov(double tol = 1E-8) {
  Eigen::MatrixXd A(5, 5);
  
  A << 10,  2,  3,  4,  5,
        6,  20,  8,  9,  1,
        1,  2, 30,  4,  5,
        6,  7,  8, 20,  0,
        1,  2,  3,  4, 10;
  
  //Compute X using solveLyapunov
  Eigen::MatrixXd X(5, 5);
  solveLyapunov(A, X);
  
  //Compute AX + XA^T - avoid lazy evaluation
  Eigen::MatrixXd AX = A * X;
  Eigen::MatrixXd XAT = X * A.transpose();
  Eigen::MatrixXd I_result = AX + XAT;
  
  //Compare for relative equality
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5, 5);
  
  if((I - I_result).norm() < tol) {
    return true;
  } else {
    return false;
  }
}

template <typename SCALAR>
struct TripletMatrix {
  std::size_t n_rows{0};  // Number of rows
  std::size_t n_cols{0};  // Number of columns
  std::vector<std::tuple<std::size_t, std::size_t, SCALAR>> triplets;
};

template <typename SCALAR>
struct CRSMatrix {
  std::size_t n_rows{0};             // Number of rows
  std::size_t n_cols{0};             // Number of columns
  std::vector<SCALAR> val;           // Value array
  std::vector<std::size_t> col_ind;  // Column indices
  std::vector<std::size_t> row_ptr;  // Row pointers
};


template<typename SCALAR>
Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>
  densify(const TripletMatrix<SCALAR>& M) {
  //Retrieve dimensions
  const int rows = M.n_rows;
  const int cols = M.n_cols;
  //Initialize matrix as dense zero matrix
  Eigen::MatrixXd result = 
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);
  //Go over the triplets and set the entries via addition
  for(auto& triplet : M.triplets) {
    int row = std::get<0>(triplet);
    int col = std::get<1>(triplet);
    double value = std::get<2>(triplet);
    
    result(row, col) += value;
  }
  return result;
}

template<typename SCALAR>
Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>
  densify(const CRSMatrix<SCALAR>& M) {
  //Retrieve dimensions
  const int rows = M.n_rows;
  const int cols = M.n_cols;
  //Initialize matrix as dense zero matrix
  Eigen::MatrixXd result = 
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);
  //Add entries from the CRS format into the dense matrix
  for(int i = 0; i < M.row_ptr.size(); ++i) {
    for(int j = M.row_ptr[i]; j < M.row_ptr[i + 1]; ++j) {
      int row = i;
      int col = M.col_ind[j];
      double value = M.val[j];
      
      result(row, col) = value;
    }
  }
  return result;
}



int main() {
  std::cout << testLyapunov() << std::endl;
}






