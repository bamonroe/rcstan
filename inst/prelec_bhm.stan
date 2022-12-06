data {
  // The number of subjects
  int<lower=0> N;
  // The number of data points per subject
  int<lower=0> T;

  // The choices
  array[N * T] int<lower=0, upper = 1> choice;

  // The probabilities
  array[N * T] real<lower = 0, upper = 1> opt1_prob1;
  array[N * T] real<lower = 0, upper = 1> opt1_prob2;
  array[N * T] real<lower = 0, upper = 1> opt1_prob3;
  array[N * T] real<lower = 0, upper = 1> opt2_prob1;
  array[N * T] real<lower = 0, upper = 1> opt2_prob2;
  array[N * T] real<lower = 0, upper = 1> opt2_prob3;

  // The outcomes
  array[N * T] real opt1_out1;
  array[N * T] real opt1_out2;
  array[N * T] real opt1_out3;
  array[N * T] real opt2_out1;
  array[N * T] real opt2_out2;
  array[N * T] real opt2_out3;

  // Max and Min outcomes with non-zero probability across the pair
  // this is for Contextual Utility
  array[N * T] real Max;
  array[N * T] real Min;
}

parameters {
  // Hyper parameters first
  // parameters for the mean of each of r, phi, eta, and mu
  real rm, pm, em, um;
  // parameters for the standard deviation of each of r, phi, eta, and mu
  real<lower = 0> rs, ps, es, us;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  array[N] real r_dist;
  array[N] real ln_phi;
  array[N] real ln_eta;
  array[N] real ln_mu;
}

model {

  // Hyper Prior Distributions
  // r mean and standard deviation
  target += normal_lpdf(rm | 0, 10);
  target += inv_gamma_lpdf(rs | 10, 3);

  // log(phi) mean and standard deviation
  target += normal_lpdf(pm | 0, 10);
  target += inv_gamma_lpdf(ps | 8, 3);

  // log(eta) mean and standard deviation
  target += normal_lpdf(em | 0, 10);
  target += inv_gamma_lpdf(es | 8, 3);

  // log(mu) mean and standard deviation
  target += normal_lpdf(um | 0, 10);
  target += inv_gamma_lpdf(us | 10, 3);

  // Variables for subject specific parameters - this should reduce memory lookups
  real r, phi, eta, mu;
  // Variable for the utility difference
  real udiff;
  // Variable keeping track of the observation
  int i;

  // Variables for probabilities, cumulative probabilities, and decision weights
  real p11, p12, p13, p21, p22, p23;
  real pw11, pw12, pw13, pw21, pw22, pw23;
  real dw11, dw12, dw13, dw21, dw22, dw23;

  // Looping through the subjects
  for (n in 1:N) {

    // CRRA prior
    target += normal_lpdf(r_dist[n] | rm, rs);
    // PWF prior
    target += normal_lpdf(ln_phi[n] | pm, ps);
    target += normal_lpdf(ln_eta[n] | em, es);
    // Fechner prior
    target += normal_lpdf(ln_mu[n] | um, us);

    // The parameters for subject "n"
    // Note for "r" that I'm taking (1 - r) right at the top here. This saves
    // (t * 8) - 1  arathmetic operations, don't undo this.
    r   = 1 - r_dist[n];
    mu  = exp(ln_mu[n]);
    phi = exp(ln_phi[n]);
    eta = exp(ln_eta[n]);

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
      if (pw12 != 1) {
        dw12 = exp(-eta * (-log(pw12))^phi);
      } else if (pw12 == 0) {
        dw12 = 0;
      } else {
        dw12 = 1;
      }
      if (pw11 != 1) {
        dw11 = exp(-eta * (-log(pw11))^phi);
      } else if (pw11 == 0) {
        dw11 = 0;
      } else {
        dw11 = 1;
      }

      dw23 = 1;  // Must always be 1 theory-wise
      if (pw22 != 1) {
        dw22 = exp(-eta * (-log(pw22))^phi);
      } else if (pw22 == 0) {
        dw22 = 0;
      } else {
        dw22 = 1;
      }
      if (pw21 != 1) {
        dw21 = exp(-eta * (-log(pw21))^phi);
      } else if (pw21 == 0) {
        dw21 = 0;
      } else {
        dw21 = 1;
      }

      // Decumulate the probabilities
      dw13 -= dw12;
      dw12 -= dw11;

      dw23 -= dw22;
      dw22 -= dw21;

      // Note that I'm NOT dividing by r, because this cancels out with Contextual utility
      // this saves us 7 arathmetic operations PER observation.
      // Utility of option 1
      udiff  = opt1_out1[i]^r * dw11 + opt1_out2[i]^r * dw12 + opt1_out3[i]^r * dw13;
      // Subtracting utility of option 2 in place
      udiff -= opt2_out1[i]^r * dw21 + opt2_out2[i]^r * dw22 + opt2_out3[i]^r * dw23;

      // Add Contextual utility
      udiff = udiff / ((Max[i]^r - Min[i]^r) * mu);

      // Logistic linking function
      udiff = 1 / (1 + exp(udiff));
      udiff = choice[i] * udiff + (1 - choice[i]) * (1 - udiff);

      // Log the likelihood and add to the posterior target
      udiff = log(udiff);

      target += udiff;
    }
  }
}
