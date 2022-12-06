.onLoad <- function(libname, pkgname) {
  # Set to maximum cores unless specified already
  options(mc.cores = getOption("mc.cores", default = parallel::detectCores()))
  # Auto write the compiled code so it doesn't need to be redone everytime
  rstan::rstan_options(auto_write = TRUE)
}
