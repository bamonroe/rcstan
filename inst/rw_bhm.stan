data {
  // The number of subjects
  int<lower=0> N;
  // The number of data points per subject
  int<lower=0> T;

  // The choices
  int<lower=0, upper = 1> choice[N * T];

  // The probabilities
  real<lower = 0, upper = 1> opt1_prob1[N * T];
  real<lower = 0, upper = 1> opt1_prob2[N * T];
  real<lower = 0, upper = 1> opt1_prob3[N * T];
  real<lower = 0, upper = 1> opt2_prob1[N * T];
  real<lower = 0, upper = 1> opt2_prob2[N * T];
  real<lower = 0, upper = 1> opt2_prob3[N * T];

  // The outcomes
  real opt1_out1[N * T];
  real opt1_out2[N * T];
  real opt1_out3[N * T];
  real opt2_out1[N * T];
  real opt2_out2[N * T];
  real opt2_out3[N * T];

  // Max and Min outcomes with non-zero probability across the pair
  // this is for Contextual Utility
  real Max[N * T];
  real Min[N * T];
}

parameters {
  // Hyper parameters first
  // parameters for the mean of each of r, phi, eta, and mu
  real rm;
  real<lower = 0> aa;
  real<lower = 0> ba;
  real<lower = 0> um;
  // parameters for the standard deviation of each of r, phi, eta, and mu
  real<lower = 0> rs;
  real<lower = 0> ab;
  real<lower = 0> bb;
  real<lower = 0> us;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  real r[N];
  real<lower = 0, upper = 1> a[N];
  real<lower = 0, upper = 1> b[N];
  real<lower = 0> mu[N];
}

model {

  // Variables for subject specific parameters - this should reduce memory lookups
  real ri;
  real ai;
  real bi;
  real mui;
  // Variable for the utility difference
  real udiff;
  // Variable keeping track of the observation
  int i;

  // Variables for probabilities, cumulative probabilities, and decision weights
  real p11;
  real p12;
  real p13;
  real p21;
  real p22;
  real p23;
  real pw11;
  real pw12;
  real pw13;
  real pw21;
  real pw22;
  real pw23;
  real dw11;
  real dw12;
  real dw13;
  real dw21;
  real dw22;
  real dw23;

  // Hyper Prior Distributions
  // r mean and standard deviation
  target += normal_lpdf(rm | 0, 10);
  target += inv_gamma_lpdf(rs | 10, 3);

  // log(gamma) mean and standard deviation
  target += inv_gamma_lpdf(aa | 0, 10);
  target += inv_gamma_lpdf(ab | 8, 3);

  // log(gamma) mean and standard deviation
  target += inv_gamma_lpdf(ba | 0, 10);
  target += inv_gamma_lpdf(bb | 8, 3);

  // log(mu) mean and standard deviation
  target += normal_lpdf(um | 0, 10);
  target += inv_gamma_lpdf(us | 10, 3);


  // Looping through the subjects
  for (n in 1:N) {

    // CRRA prior
    target += normal_lpdf(r[n] | rm, rs);
    // PWF prior
    target += beta_lpdf(a[n] | aa, ab);
    target += beta_lpdf(b[n] | ba, bb);
    // Fechner prior
    target += normal_lpdf(ln_mu[n] | um, us);

    // The parameters for subject "n"
    // Note for "r" that I'm taking (1 - r) right at the top here. This saves
    // (t * 8) - 1  arathmetic operations, don't undo this.
    ri   = 1 - r[n];
    ai   = a[n];
    bi   = b[n];
    mui  = mu[n];

    // TODO:
    // Still need to add jacobian correction for variable transformations, or
    // make use of Stan's built-in variable transformation functionality.

    // Looping through each of the observations per-subject
    for (t in 1:T) {
      i = n * t;

      // Some short-hand reference for the probabilities
      p11 = opt1_prob1[i];
      p12 = opt1_prob2[i];
      p13 = opt1_prob3[i];
      p21 = opt2_prob1[i];
      p22 = opt2_prob2[i];
      p23 = opt2_prob3[i];

      // Cumulate the probabilities from higest to lowest
      pw11 = p11;
      pw12 = p12 + p11;
      pw13 = 1;

      pw21 = p21;
      pw22 = p22 + p21;
      pw23 = 1;

      // Add the probability weighting function
      dw13 = 1;  // Must always be 1 theory-wise
      dw12 = pw12 + (pw12^3 - (ai + 1) * pw12^2 + ai * pw12) * (3 - 3bi) / (ai^2 - ai + 1)
      dw11 = pw11 + (pw11^3 - (ai + 1) * pw11^2 + ai * pw11) * (3 - 3bi) / (ai^2 - ai + 1)

      dw23 = 1;  // Must always be 1 theory-wise
      dw22 = pw22 + (pw22^3 - (ai + 1) * pw22^2 + ai * pw22) * (3 - 3bi) / (ai^2 - ai + 1)
      dw21 = pw21 + (pw21^3 - (ai + 1) * pw21^2 + ai * pw21) * (3 - 3bi) / (ai^2 - ai + 1)

      // Decumulate the probabilities
      dw13 -= dw12;
      dw12 -= dw11;

      dw23 -= dw22;
      dw22 -= dw21;

      // Note that I'm NOT dividing by r, because this cancels out with Contextual utility
      // this saves us 7 arathmetic operations PER observation.
      // Utility of option 1
      udiff  = opt1_out1[i]^ri * dw11 + opt1_out2[i]^ri * dw12 + opt1_out3[i]^ri * dw13;
      // Subtracting utility of option 2 in place
      udiff -= opt2_out1[i]^ri * dw21 + opt2_out2[i]^ri * dw22 + opt2_out3[i]^ri * dw23;

      // Add Contextual utility
      udiff = udiff / ((Max[i]^ri - Min[i]^ri) * mu);

      // Logistic linking function
      udiff = 1 / (1 + exp(udiff));
      udiff = choice[i] * udiff + (1 - choice[i]) * (1 - udiff);

      // Log the likelihood and add to the posterior target
      udiff = log(udiff);

      target += udiff;
    }
  }
}
