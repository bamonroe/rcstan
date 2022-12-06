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

  array[N * T] real Max;
  array[N * T] real Min;
}

parameters {
  real r, ln_phi, ln_eta, ln_mu;
}

model {

  target += normal_lpdf(r | 0, 10);
  target += normal_lpdf(ln_phi | 0, 5);
  target += normal_lpdf(ln_eta | 0, 5);
  target += normal_lpdf(ln_mu  | -2, 2);

  real phi = exp(ln_phi);
  real eta = exp(ln_eta);
  real mu  = exp(ln_mu);

  real udiff;
  int i;

  real dw11, dw12, dw13, dw21, dw22, dw23;
  real p11, p12, p13, p21, p22, p23;

  real tt = 0;

  // Looping through the subjects
  for (n in 1:N) {
    // Looping through each of the observations per-subject
    for (t in 1:T) {
      i = n * t;

      p11 = opt1_prob1[i];
      p12 = opt1_prob2[i];
      p13 = opt1_prob3[i];
      p21 = opt2_prob1[i];
      p22 = opt2_prob2[i];
      p23 = opt2_prob3[i];

      dw12 = p12 + p13;
      dw13 = p13;

      dw22 = p22 + p23;
      dw23 = p23;

      // Add the probability weighting function
      dw11 = 1;  // Must always be 1 theory-wise
      dw12 = exp(-eta * (-log(dw12))^phi);
      dw13 = exp(-eta * (-log(dw13))^phi);

      dw21 = 1;  // Must always be 1 theory-wise
      dw22 = exp(-eta * (-log(dw22))^phi);
      dw23 = exp(-eta * (-log(dw23))^phi);

      // Decumulate the probabilities
      dw11 -= dw12;
      dw12 -= dw13;

      dw21 -= dw22;
      dw22 -= dw23;

      if (p11 == 0) dw11 = 0;
      if (p12 == 0) dw12 = 0;
      if (p13 == 0) dw13 = 0;

      if (p21 == 0) dw21 = 0;
      if (p22 == 0) dw22 = 0;
      if (p23 == 0) dw23 = 0;

      if (p11 == 1) dw11 = 1;
      if (p12 == 1) dw12 = 1;
      if (p13 == 1) dw13 = 1;

      if (p21 == 1) dw21 = 1;
      if (p22 == 1) dw22 = 1;
      if (p23 == 1) dw23 = 1;


      udiff  = opt1_out1[i]^r * dw11 + opt1_out2[i]^r * dw12 + opt1_out3[i]^r * dw13;
      udiff -= opt2_out1[i]^r * dw21 + opt2_out2[i]^r * dw22 + opt2_out3[i]^r * dw23;

      udiff = udiff / ((Max[i]^r - Min[i]^r) * mu);

      udiff = 1 / (1 + exp(udiff));

      udiff = choice[i] * udiff + (1 - choice[i]) * (1 - udiff);
      udiff = log(udiff);

      target += udiff;
    }
  }

}
