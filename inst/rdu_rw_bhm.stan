data {
  // The number of subjects
  int<lower=1> N;
  // The number of data points by subject
  int<lower=1> T[N];
  // The number of rows of data total
  int<lower=1> ndat;
  // The number of options in the dataset
  int<lower=2> nopts;
  // The number of outcomes in the dataset
  int<lower=2> nouts;
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
  matrix[ndat, nouts] probs1;
  matrix[ndat, nouts] probs2;

  // The outcomes
  matrix[ndat, nouts] outs1;
  matrix[ndat, nouts] outs2;

  // Max and Min outcomes with non-zero probability across the pair
  // this is for Contextual Utility
  real Max[ndat];
  real Min[ndat];
}

transformed data {

  // Cumulative probabilities from highest to lowest here.
  // Doing this operation here allows it to only be calculated once instead of
  // every time the model is called

  // Initialize the decision weights at the value of the first probability, the
  // highest outcome
  matrix[ndat, nouts] cprob1 = rep_matrix(probs1[, 1], nouts);
  matrix[ndat, nouts] cprob2 = rep_matrix(probs2[, 1], nouts);

  for (kk in 2:nouts) {
    cprob1[, kk] = cprob1[, kk - 1] + probs1[, kk];
    cprob2[, kk] = cprob2[, kk - 1] + probs2[, kk];
  }
  // Normalize back so they add up to one
  cprob1 = cprob1 ./ rep_matrix(cprob1[, nouts], nouts);
  cprob2 = cprob2 ./ rep_matrix(cprob2[, nouts], nouts);

  // Indicator variables for whether a cumulative probability is zero
  // We need these because the Prelec weighting function is not defined at zero
  matrix[ndat, nouts] cprob1_c;
  matrix[ndat, nouts] cprob2_c;
  for (nn in 1:ndat) {
    for (kk in 1:nouts) {
      cprob1_c[nn, kk] = cprob1[nn, kk] == 0 ? 1 : 0;
      cprob2_c[nn, kk] = cprob2[nn, kk] == 0 ? 1 : 0;
    }
  }
  // Replace the zeros with small numbers.
  cprob1 = (1.0 - cprob1_c) .* cprob1 + cprob1_c * 0.05;
  cprob2 = (1.0 - cprob2_c) .* cprob2 + cprob2_c * 0.05;

  // We need the number of outcomes minus 1 several times
  int nouts_m1 = nouts - 1;

}

parameters {
  // Hyper parameters first
  // parameters for the mean of each of r, a, b, and mu
  real hyper_r_mean;
  real hyper_a_mean;
  real hyper_c_mean;
  real hyper_mu_mean;
  // parameters (proportional to) the standard deviation of each of r, phi, eta, and mu
  real hyper_r_lnsd;
  real hyper_a_lnsd;
  real hyper_c_lnsd;
  real hyper_mu_lnsd;

  // Arrays of parameters for each subject. Each arrary keeps N parameters, N
  // being the number of subjects
  real r[N];
  real a[N];
  real c[N];
  real ln_mu[N];

  // Vector for the parameters that defined the covariate effects
  real dem[ncovar_est];
}

model {
  // Variables for subject specific parameters - this should reduce memory lookups
  real apar;
  real bpar;

  real ri;
  real ai;
  real bi;
  real ci;
  real mu;
  // Variables for covariate-corrected hyper-parameters. You need to change
  // the index manually per-model
  real hyper[8];
  // Variable for the utility difference
  real udiff;
  // Variables for utilities
  real u1;
  real u2;
  real u_max;
  real u_min;
  // Variable keeping track of covariate effects
  int di = 0;
  // Variable keeping track of the observation
  int i = 0;
  // Variable for use in loops
  int j = 0;

  // Variables for the calculated decision weights
  vector[nouts] dw1;
  vector[nouts] dw2;

  // Hyper Prior Distributions
  // r mean and standard deviation
  target += normal_lpdf(hyper_r_mean | 0, 100);
  target += normal_lpdf(hyper_r_lnsd | 0, 100);

  // mu and phi for the a distribution
  target += normal_lpdf(hyper_a_mean | 0, 100);
  target += normal_lpdf(hyper_a_lnsd | 0, 100);

  // mu and phi for the b distribution
  target += normal_lpdf(hyper_c_mean | 0, 100);
  target += normal_lpdf(hyper_c_lnsd | 0, 100);

  // log(mu) mean and standard deviation
  target += normal_lpdf(hyper_mu_mean | 0, 100);
  target += normal_lpdf(hyper_mu_lnsd | 0, 100);

  // Add the prior for each possible covar effect
  // For now, a weak prior on 0. Putting a stronger prior on 0 requires more
  // evidence to infer that an effect is really there.
  for (d in 1:ncovar_est) {
    target += normal_lpdf(dem[d] | 0, 100);
  }

  // Looping through the subjects
  for (n in 1:N) {
    // Set the vector for the covariate-corrected hyper-parameters equal to the
    // base hyper-parameters
    hyper[1] = hyper_r_mean;
    hyper[2] = hyper_r_lnsd;
    hyper[3] = hyper_a_mean;
    hyper[4] = hyper_a_lnsd;
    hyper[5] = hyper_c_mean;
    hyper[6] = hyper_c_lnsd;
    hyper[7] = hyper_mu_mean;
    hyper[8] = hyper_mu_lnsd;

    // Reset the effect counter to 0
    di = 0;
    // Note that we're cycling through 4 posisble hyper-parameters. This is
    // model dependent, and up to the user to change
    for (h in 1:8) {
      // We're only doing this if the hyper-parameter is being used
      if (nhyper >= h) {
        // Loop through each possible covar to see if it's being applied
        for (d in 1:ncvars) {
          // If it is, increment the effect counter, and apply the effect to the hyper-parameter
          if (cvarmap[d, h] == 1) {
            di += 1;
            hyper[h] += covars[n, d] * dem[di];
          }
        }
      }
    }

    // Put the hyper-parameters into their correct limit
    hyper[2] = exp(hyper[2]);
    hyper[4] = exp(hyper[4]);
    hyper[6] = exp(hyper[6]);
    hyper[8] = exp(hyper[8]);

    // Priors for the parameters
    target += normal_lpdf(r[n]     | hyper[1], hyper[2]);
    target += normal_lpdf(a[n]     | hyper[3], hyper[4]);
    target += normal_lpdf(c[n]     | hyper[5], hyper[6]);
    target += normal_lpdf(ln_mu[n] | hyper[7], hyper[8]);

    // The parameters for subject "n"
    ri  = r[n];
    ai  = 1 / (1 + exp(-a[n]));
    ci  = 1 / (1 + exp(-c[n]));

    // To allow convex, then concave, shapes, "bi" must be allowed to be
    // greater than 1, but it still must be less than the second term in the
    // following. Since draws of ci are from a logit-normal distribution, we can
    // multiply ci by this maximum to get a new term that is between 0 and the
    // maximum
    bi = ci * (1 + 1.0/3.0 * ((ai^2 - ai + 1)/(.5 + ((ai - .5)^2)^.5)));

    mu = exp(ln_mu[n]);

    // Looping through each of the observations per-subject
    for (t in 1:T[n]) {
      i += 1;

      // Add the probability weighting function
      for (out in 1:nouts_m1) {
        dw1[out] = cprob1[i, out] + (cprob1[i, out]^3 - (ai + 1) * cprob1[i, out]^2 + ai * cprob1[i, out]) * (3 - 3 * bi) / (ai^2 - ai + 1);
        dw2[out] = cprob2[i, out] + (cprob2[i, out]^3 - (ai + 1) * cprob2[i, out]^2 + ai * cprob2[i, out]) * (3 - 3 * bi) / (ai^2 - ai + 1);
      }
      dw1[nouts] = 1.0;
      dw2[nouts] = 1.0;

      // If the values were actually zeros, add the zeros back in
      for (out in 1:nouts) {
        dw1[out] = (1 - cprob1_c[i, out]) * dw1[out];
        dw2[out] = (1 - cprob2_c[i, out]) * dw2[out];
      }

      // Decumulate the probabilities
      for (out in 1:nouts_m1) {
        // Stan only does incremental loops, but we need decrement here
        j = (nouts - 1) - out + 1;

        dw1[j + 1] -= dw1[j];
        dw2[j + 1] -= dw2[j];
      }

      // Calculate the EUT and RDU of the options
      u1 = 0;
      u2 = 0;
      for (out in 1:nouts) {
        // RDU uses dw
        u1 += (dw1[out] * outs1[i, out]^(1 - ri) / (1 - ri));
        u2 += (dw2[out] * outs2[i, out]^(1 - ri) / (1 - ri));
      }

      // RDU utility difference using contextual utility
      u_max = Max[i]^(1 - ri) / (1 - ri);
      u_min = Min[i]^(1 - ri) / (1 - ri);
      udiff = (u1 - u2) / (u_max - u_min);
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
