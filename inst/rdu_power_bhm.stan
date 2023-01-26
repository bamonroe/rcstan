data {
  // The number of subjects
  int<lower=0> N;
  // The number of data points by subject
  int<lower=0> T[N];
  // The number of rows of data total
  int<lower=0> ndat;

  // The choices
  int<lower=0, upper = 1> choice[ndat];

  // The probabilities
  real<lower = 0, upper = 1> opt1_prob1[ndat];
  real<lower = 0, upper = 1> opt1_prob2[ndat];
  real<lower = 0, upper = 1> opt1_prob3[ndat];
  real<lower = 0, upper = 1> opt1_prob4[ndat];

  real<lower = 0, upper = 1> opt2_prob1[ndat];
  real<lower = 0, upper = 1> opt2_prob2[ndat];
  real<lower = 0, upper = 1> opt2_prob3[ndat];
  real<lower = 0, upper = 1> opt2_prob4[ndat];

  // The outcomes
  real opt1_out1[ndat];
  real opt1_out2[ndat];
  real opt1_out3[ndat];
  real opt1_out4[ndat];

  real opt2_out1[ndat];
  real opt2_out2[ndat];
  real opt2_out3[ndat];
  real opt2_out4[ndat];

  // Max and Min outcomes with non-zero probability across the pair
  // this is for Contextual Utility
  real Max[ndat];
  real Min[ndat];
}

parameters {
  // Hyper parameters first
  // parameters for the mean of each of r, phi, eta, and mu
  real rm;
  real gm;
  real um;
  // parameters for the standard deviation of each of r, phi, eta, and mu
  real<lower = 0> rs;
  real<lower = 0> gs;
  real<lower = 0> us;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  real r[N];
  real ln_gamma[N];
  real ln_mu[N];
}

model {
  // Variables for subject specific parameters - this should reduce memory lookups
  real ri;
  real g;
  real mu;
  // Variable for the utility difference
  real udiff;
  // Variable keeping track of the observation
  int i = 0;

  // Variables for probabilities
  real p11;
  real p12;
  real p13;
  real p14;

  real p21;
  real p22;
  real p23;
  real p24;

  real pw11;
  real pw12;
  real pw13;
  real pw14;

  real pw21;
  real pw22;
  real pw23;
  real pw24;

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

  // log(gamma) mean and standard deviation
  target += normal_lpdf(gm | 0, 10);
  target += inv_gamma_lpdf(gs | 8, 3);

  // log(mu) mean and standard deviation
  target += normal_lpdf(um | 0, 10);
  target += inv_gamma_lpdf(us | .001, .001);

  // Looping through the subjects
  for (n in 1:N) {

    // CRRA prior
    target += normal_lpdf(r[n] | rm, rs);
    // PWF prior
    target += normal_lpdf(ln_gamma[n] | gm, gs);
    // Fechner prior
    target += normal_lpdf(ln_mu[n] | um, us);

    // The parameters for subject "n"
    ri  = r[n];
    g   = exp(ln_gamma[n]);
    mu  = exp(ln_mu[n]);

    // Looping through each of the observations per-subject
    for (t in 1:T[n]) {
      i += 1;

      // Some short-hand reference for the probabilities
      p11 = opt1_prob1[i];
      p12 = opt1_prob2[i];
      p13 = opt1_prob3[i];
      p14 = opt1_prob4[i];

      p21 = opt2_prob1[i];
      p22 = opt2_prob2[i];
      p23 = opt2_prob3[i];
      p24 = opt2_prob4[i];

      // Cumulate the probabilities from higest to lowest
      pw11 = p11;
      pw12 = p12 + p11;
      pw13 = p13 + p12;
      pw14 = 1;

      pw21 = p21;
      pw22 = p22 + p21;
      pw23 = p23 + p22;
      pw24 = 1;

      // Add the probability weighting function
      dw14 = 1;  // Must always be 1 theory-wise
      dw13 = pw13^g;
      dw12 = pw12^g;
      dw11 = pw11^g;

      dw24 = 1;  // Must always be 1 theory-wise
      dw23 = pw23^g;
      dw22 = pw22^g;
      dw21 = pw21^g;

      // Decumulate the probabilities
      dw14 -= dw13;
      dw13 -= dw12;
      dw12 -= dw11;

      dw24 -= dw23;
      dw23 -= dw22;
      dw22 -= dw21;

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
