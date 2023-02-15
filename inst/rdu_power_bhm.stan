data {
  // The number of subjects
  int<lower=0> N;
  // The number of data points by subject
  int<lower=0> T[N];
  // The number of rows of data total
  int<lower=0> ndat;
  // The number of covariate effects to estimate
  int<lower=0> ncovar_est;
  // The number of unique covariates passed across all hyper parameters
  int<lower=0> ncvars;
  // Number of hyper-parameter
  int<lower=0> nhyper;
  // The list of bools for each possible covar, for each hyper-parameter
  int<lower=0> cvarmap[ncvars, nhyper];
  // This is the matrix of all unique covars, with 1 row per subjects, 1 column
  // for each possible covar across all hyper-parameters
  real<lower=0> covars[N, ncvars];

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
  //
 
  // Hyper parameters first
  // parameters for the CRRA parameter
  real hyper_r_mean;
  real hyper_r_lnsd;
  // parameters for the PWF gamma term
  real hyper_g_mean;
  real hyper_g_lnsd;
  // parameters for the Fechner term
  real hyper_u_mean;
  real hyper_u_lnsd;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  real r[N];
  real ln_gamma[N];
  real ln_mu[N];

  // Vector for the parameters that defined the covariate effects
  real dem[ncovar_est];
}

model {
  // Variables for subject specific parameters - this should reduce memory lookups
  real ri;
  real g;
  real mu;
  // Variables for covariate-corrected hyper-parameters. You need to change
  // the index manually per-model
  real hyper[6];
  // Variable for the utility difference
  real udiff;
  // Variable keeping track of covariate effects
  int ci = 0;
  // Variable keeping track of the observation
  int i = 0;

  // Variables for decision weights
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
  target += normal_lpdf(hyper_r_mean | 0, 100);
  target += normal_lpdf(hyper_r_lnsd | 0, 100);

  // log(gamma) mean and standard deviation
  target += normal_lpdf(hyper_g_mean | 0, 100);
  target += normal_lpdf(hyper_g_lnsd | 0, 100);

  // log(mu) mean and standard deviation
  target += normal_lpdf(hyper_u_mean | 0, 100);
  target += normal_lpdf(hyper_u_lnsd | 0, 100);

  // Add the prior for each possible covar effect
  // For now, a weak prior on 0. Putting a stronger prior on 0 requires more
  // evidence to infer that an effect is really there.
  for (c in 1:ncovar_est) {
    target += normal_lpdf(dem[c] | 0, 100);
  }

  // Looping through the subjects
  for (n in 1:N) {
    // Set the vector for the covariate-corrected hyper-parameters equal to the
    // base hyper-parameters
    hyper[1] = hyper_r_mean;
    hyper[2] = hyper_r_lnsd;
    hyper[3] = hyper_g_mean;
    hyper[4] = hyper_g_lnsd;
    hyper[5] = hyper_u_mean;
    hyper[6] = hyper_u_lnsd;

    // Reset the effect counter to 0
    ci = 0;
    // Note that we're cycling through 4 posisble hyper-parameters. This is
    // model dependent, and up to the user to change
    for (h in 1:6) {
      // We're only doing this if the hyper-parameter is being used
      if (nhyper >= h) {
        // Loop through each possible covar to see if it's being applied
        for (c in 1:ncvars) {
          // If it is, increment the effect counter, and apply the effect to the hyper-parameter
          if (cvarmap[c, h] == 1) {
            ci += 1;
            hyper[h] += covars[n, c] * dem[ci];
          }
        }
      }
    }

    // Put the hyper-parameters into their correct limit
    hyper[2] = exp(hyper[2]);
    hyper[4] = exp(hyper[4]);
    hyper[6] = exp(hyper[6]);

    // CRRA prior
    target += normal_lpdf(r[n] | hyper[1], hyper[2]);
    // PWF prior
    target += normal_lpdf(ln_gamma[n] | hyper[3], hyper[4]);
    // Fechner prior
    target += normal_lpdf(ln_mu[n] | hyper[5], hyper[6]);

    // The parameters for subject "n"
    ri  = r[n];
    g   = exp(ln_gamma[n]);
    mu  = exp(ln_mu[n]);

    // Looping through each of the observations per-subject
    for (t in 1:T[n]) {
      i += 1;

      // Cumulate the probabilities from higest to lowest
      dw11 = opt1_prob1[i];
      dw12 = dw11 + opt1_prob2[i];
      dw13 = dw12 + opt1_prob3[i];
      dw14 = dw13 + opt1_prob4[i];

      dw21 = opt2_prob1[i];
      dw22 = dw21 + opt2_prob2[i];
      dw23 = dw22 + opt2_prob3[i];
      dw24 = dw23 + opt2_prob4[i];

      dw11 = dw11 / dw14;
      dw12 = dw12 / dw14;
      dw13 = dw13 / dw14;
      dw14 = dw14 / dw14;
      dw21 = dw21 / dw24;
      dw22 = dw22 / dw24;
      dw23 = dw23 / dw24;
      dw24 = dw24 / dw24;

      // Add the probability weighting function
      dw11 = dw11^g;
      dw12 = dw12^g;
      dw13 = dw13^g;
      dw14 = 1;  // Must always be 1 theory-wise

      dw21 = dw21^g;
      dw22 = dw22^g;
      dw23 = dw23^g;
      dw24 = 1;  // Must always be 1 theory-wise

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
