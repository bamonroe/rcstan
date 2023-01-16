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
  real<lower = 0, upper = 1> opt1_prob4[N * T];

  real<lower = 0, upper = 1> opt2_prob1[N * T];
  real<lower = 0, upper = 1> opt2_prob2[N * T];
  real<lower = 0, upper = 1> opt2_prob3[N * T];
  real<lower = 0, upper = 1> opt2_prob4[N * T];

  // The outcomes
  real opt1_out1[N * T];
  real opt1_out2[N * T];
  real opt1_out3[N * T];
  real opt1_out4[N * T];

  real opt2_out1[N * T];
  real opt2_out2[N * T];
  real opt2_out3[N * T];
  real opt2_out4[N * T];

  // Max and Min outcomes with non-zero probability across the pair
  // this is for Contextual Utility
  real Max[N * T];
  real Min[N * T];
}

parameters {
  // Hyper parameters first
  // parameters for the mean of each of r, and mu
  real rm;
  real um;
  // parameters for the standard deviation of each of r, phi, eta, and mu
  real<lower = 0> rs;
  real<lower = 0> us;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  real r[N];
  real ln_mu[N];
}

model {
  // Variables for subject specific parameters - this should reduce memory lookups
  real ri;
  real mu;
  // Variable for the utility difference
  real udiff;
  // Variable keeping track of the observation
  int i;

  // Variables for probabilities
  real dw11;
  real dw12;
  real dw13;
  real dw14;

  real dw21;
  real dw22;
  real dw23;
  real dw24;

  // Hyper Prior Distributions
  // r mean and standard deviation
  target += normal_lpdf(rm | 0, 10);
  target += inv_gamma_lpdf(rs | .001, .001);

  // log(mu) mean and standard deviation
  target += normal_lpdf(um | 0, 10);
  target += inv_gamma_lpdf(us | .001, .001);

  // Looping through the subjects
  for (n in 1:N) {

    // CRRA prior
    target += normal_lpdf(r[n] | rm, rs);
    // Fechner prior
    target += normal_lpdf(ln_mu[n] | um, us);

    // The parameters for subject "n"
    ri  = r[n];
    mu  = exp(ln_mu[n]);

    // TODO:
    // Still need to add jacobian correction for variable transformations, or
    // make use of Stan's built-in variable transformation functionality.

    // Looping through each of the observations per-subject
    for (t in 1:T) {
      i = n * t;

      // Some short-hand reference for the probabilities
      dw11 = opt1_prob1[i];
      dw12 = opt1_prob2[i];
      dw13 = opt1_prob3[i];
      dw14 = opt1_prob4[i];

      dw21 = opt2_prob1[i];
      dw22 = opt2_prob2[i];
      dw23 = opt2_prob3[i];
      dw24 = opt2_prob4[i];

      // Note that I'm NOT dividing by r, because this cancels out with Contextual utility
      // this saves us 7 arithmetic operations PER observation.
      // Utility of option 1
      udiff  = opt1_out1[i]^(1 - ri) / (1 - ri) * dw11;
      udiff += opt1_out2[i]^(1 - ri) / (1 - ri) * dw12;
      udiff += opt1_out3[i]^(1 - ri) / (1 - ri) * dw13;
      udiff += opt1_out4[i]^(1 - ri) / (1 - ri) * dw14;

      // Subtracting utility of option 2 in place
      udiff -= opt2_out1[i]^(1 - ri) / (1 - ri) * dw21;
      udiff -= opt2_out2[i]^(1 - ri) / (1 - ri) * dw22;
      udiff -= opt2_out3[i]^(1 - ri) / (1 - ri) * dw23;
      udiff -= opt2_out4[i]^(1 - ri) / (1 - ri) * dw24;

      // Add Contextual utility
      udiff = udiff / (Max[i]^(1 - ri) / (1 - ri) - Min[i]^(1 - ri) / (1 - ri));
      udiff = udiff / mu;

      // Logistic linking function
      udiff = 1 / (1 + exp(udiff));
      udiff = choice[i] * udiff + (1 - choice[i]) * (1 - udiff);

      // Log the likelihood and add to the posterior target
      udiff = log(udiff);

      target += udiff;
    }
  }
}
