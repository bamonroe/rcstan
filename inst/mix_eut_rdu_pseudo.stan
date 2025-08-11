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
  matrix<lower = 0, upper = 1> probs1[ndat, nouts]
  matrix<lower = 0, upper = 1> probs2[ndat, nouts]

  // The outcomes
  matrix outs1[ndat, nouts]
  matrix outs2[ndat, nouts]

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

}

parameters {
  // This is a Cumulative Prospect Theory model. For a single agent, this
  // assumes different utility and probability weighting functions for gain and
  // loss domain outcomes. Thus there are 2 CRRA coefficients, 4 PWF parameters,
  // but only 1 Fechner term, which we assume is constant across both gains and
  // losses.
  //
  // As noted in detail by Harrison & Swarthout (2023), a requirement for this
  // structure is that the data contain gain, loss, and, critically, mixed
  // framed lotteries. We don't do data checks in this code, it is incumbent on
  // the user.
  //
  // There are multiple restrictions that can be put in place at the hierarchy level:
  // The gain/loss parameters could have the same prior mean (seems too restrictive)
  // The gain/loss parameters could have the same prior standard deviation (more plausible)
  //
  // We start with no such restrictions, treating CPT as more or less a spline
  // of two RDU utility functions knotted at x = 0.
  //
  // Hyper parameters first
  //

  // Hyper parameters first
  // parameters for the CRRA parameter
  real hyper_r_eut_mean;
  real hyper_r_eut_lnsd;

  // parameters for the CRRA parameter
  real hyper_r_rdu_mean;
  real hyper_r_rdu_lnsd;
  // parameters for the PWF term
  real hyper_phi_rdu_mean;
  real hyper_phi_rdu_lnsd;

  real hyper_eta_rdu_mean;
  real hyper_eta_rdu_lnsd;

  // parameters for the Fechner term
  real hyper_u_mean;
  real hyper_u_lnsd;

  // parameters for the mixture term
  real hyper_w_mean;
  real<lower = .2, upper = 2.5> hyper_w_sd;

  // Arrays of parameters for each subject. Each array keeps N parameters, N
  // being the number of subjects
  real r_eut[N];

  real r_rdu[N];
  real ln_phi_rdu[N];
  real ln_eta_rdu[N];

  real ln_mu[N];

  real ln_w[N];

  // Vector for the parameters that defined the covariate effects
  real dem[ncovar_est];
}


model {
  // Variables for subject specific parameters - this should reduce memory lookups
  real ri_eut;

  real ri_rdu;
  real phi_rdu;
  real eta_rdu;

  real u_max;
  real u_min;

  real mu;
  real w;

  // Variables for covariate-corrected hyper-parameters. You need to change
  // the index manually per-model
  real hyper[12];
  // Variable for the utility difference
  real udiff_eut;
  real udiff_rdu;
  // Variable keeping track of covariate effects
  int ci = 0;
  // Variable keeping track of the observation
  int i = 0;
  // Variable for use in loops
  int j = 0;

  // Variables for the calculated decision weights
  vector dw1[nouts];
  vector dw2[nouts];

  // Variables to check if outcomes are positive or negative
  real u1_eut;
  real u2_eut;

  real u1_rdu;
  real u2_rdu;

  // Hyper Prior Distributions
  // r mean and standard deviation
  target += normal_lpdf(hyper_r_eut_mean | 0, 100);
  target += normal_lpdf(hyper_r_eut_lnsd | 0, 100);
  target += normal_lpdf(hyper_r_rdu_mean | 0, 100);
  target += normal_lpdf(hyper_r_rdu_lnsd | 0, 100);

  // log(phi) mean and standard deviation
  target += normal_lpdf(hyper_phi_rdu_mean | 0, 100);
  target += normal_lpdf(hyper_phi_rdu_lnsd | 0, 100);

  // log(eta) mean and standard deviation
  target += normal_lpdf(hyper_eta_rdu_mean | 0, 100);
  target += normal_lpdf(hyper_eta_rdu_lnsd | 0, 100);

  // log(mu) mean and standard deviation
  target += normal_lpdf(hyper_u_mean | 0, 100);
  target += normal_lpdf(hyper_u_lnsd | 0, 100);

  // log(w) mean and standard deviation
  // target += uniform_lpdf(hyper_w_mean | -2, 2);
  // target += uniform_lpdf(hyper_w_sd   | .2, 2.5);
  target += uniform_lpdf(hyper_w_mean | -0.05, 0.05);
  target += uniform_lpdf(hyper_w_sd   | .2, 2.5);

  // Looping through the subjects
  for (n in 1:N) {

    // CRRA prior
    target += normal_lpdf(r_eut[n] | hyper_r_eut_mean, exp(hyper_r_eut_lnsd));
    target += normal_lpdf(r_rdu[n] | hyper_r_rdu_mean, exp(hyper_r_rdu_lnsd));
    // PWF prior
    target += normal_lpdf(ln_phi_rdu[n] | hyper_phi_rdu_mean, exp(hyper_phi_rdu_lnsd));
    target += normal_lpdf(ln_eta_rdu[n] | hyper_eta_rdu_mean, exp(hyper_eta_rdu_lnsd));

    // Fechner prior
    target += normal_lpdf(ln_mu[n] | hyper_u_mean, exp(hyper_u_lnsd));

    // Mixture prior
    target += normal_lpdf(ln_w[n] | hyper_w_mean, hyper_w_sd);

    // The parameters for subject "n"
    // This restricts the value of r < 1
    ri_eut  = r_eut[n];
    ri_rdu  = r_rdu[n];

    phi_rdu = exp(ln_phi_rdu[n]);
    eta_rdu = exp(ln_eta_rdu[n]);

    mu = exp(ln_mu[n]);
    w  = 1 / (1 + exp(-ln_w[n]));

    // Looping through each of the observations per-subject
    for (t in 1:T[n]) {
      i += 1;

      // Add the probability weighting function
      for (out in 1:(nouts - 1)) {
        dw1[out] = exp(-eta_rdu * (-log(cprob1[i, out]))^phi_rdu);
        dw2[out] = exp(-eta_rdu * (-log(cprob2[i, out]))^phi_rdu);
      }
      dw1[nouts] = 1.0;
      dw2[nouts] = 1.0;

      // If the values were actually zeros, add the zeros back in
      for (out in 1:nouts) {
        dw1[out] = (1 - cprob1_c[i, out]) * dw1[out];
        dw2[out] = (1 - cprob2_c[i, out]) * dw2[out];
      }

      // Decumulate the probabilities
      for (out in 1:(nouts - 1) {
        // Stan only does incremental loops, but we need decrement here
        j = (nouts - 1) - out + 1;

        dw1[j + 1] -= dw1[j];
        dw2[j + 1] -= dw2[j];
      }

      // Calculate the EUT and RDU of the options
      u1_eut = 0;
      u1_rdu = 0;
      for (out in 1:nouts) {
        // EUT uses probs
        u1_eut += (probs1[i, out] * outs1[i, out]^(1 - ri_eut) / (1 - ri_eut));
        u2_eut += (probs2[i, out] * outs2[i, out]^(1 - ri_eut) / (1 - ri_eut));
        // RDU uses dw
        u1_rdu += (dw1[out] * outs1[i, out]^(1 - ri_rdu) / (1 - ri_rdu));
        u2_rdu += (dw2[out] * outs2[i, out]^(1 - ri_rdu) / (1 - ri_rdu));
      }

      // EUT utility difference using contextual utility
      u_max = Max[i]^(1 - ri_eut) / (1 - ri_eut);
      u_min = Min[i]^(1 - ri_eut) / (1 - ri_eut);
      udiff_eut = (u1_eut - u2_eut) / ((u_max - u_min) * mu);

      // RDU utility difference using contextual utility
      u_max = Max[i]^(1 - ri_rdu) / (1 - ri_rdu);
      u_min = Min[i]^(1 - ri_rdu) / (1 - ri_rdu);
      udiff_rdu = (u1_rdu - u2_rdu) / ((u_max - u_min) * mu);

      // Logistic linking function
      udiff_eut = 1 / (1 + exp(udiff_eut));
      udiff_eut = choice[i] * udiff_eut + (1 - choice[i]) * (1 - udiff_eut);

      udiff_rdu = 1 / (1 + exp(udiff_rdu));
      udiff_rdu = choice[i] * udiff_rdu + (1 - choice[i]) * (1 - udiff_rdu);

      // Log the likelihood and add to the posterior target
      target += log(w * udiff_eut + (1 - w) * udiff_rdu);
    }
  }
}


