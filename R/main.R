mod_map <- function(i) {
  c(
    "eut_bhm",
    "rdu_power_bhm",
    "rdu_rw_bhm"
  )[i]
}

#' @title Get the raw stan code
#' @param mod the name of the PWF function
#' @param bhm boolean, TRUE indicating to use the BHM model
#' @export
get_stan_code <- function(mod_num = 1) {
  fname <- paste0(mod_map(mod_num), ".stan")
  out <- system.file(package = "rcstan", fname)
  out
}

# The data needs some checks to make sure it'll run
check_dat <- function(dat) {

  # Is a data.frame
  if (! is.data.frame(dat)) {
    stop("The passed data needs to be a data.frame")
  }

  # Required columns
  cn <- colnames(dat)
  rcn <- c("ID", "choice")
  not_present <- ! rcn %in% cn
  if (any(not_present)) {
    not_present <- paste(not_present, collapse = ", ")
    not_present <- paste0("The following columns are required but not present: ", not_present)
    stop(not_present)
  }

}


#' @title Esimate the Stan model
#' @param dat the name of the stan file
#' @export
run_stan <- function(dat, mod_num = 1, stan_opts = list()) {

  # Make sure the dat passes the checks
  check_dat(dat)

  # Get the file for the stan code
  fname <- get_stan_code(mod_num)

  # The number of subjects in this dataset
  nsubs <- length(unique(dat$ID))

  # The number of observations per subject
  ntasks <- unlist(lapply(split(dat, dat$ID), nrow))

  cnames <- colnames(dat)
  # Figure out the number of unique options by parsing the column names
  nopts <- gsub("opt([0-9]+)_out[0-9]+", "\\1", cnames, perl = TRUE)
  uopts <- sort(unique(nopts))
  nopts <- length(uopts)

  # Figure ou the number of prizes associated with each lottery (to be the same
  # for every lottery)
  nouts <- paste0("opt", uopts[1], "_out([0-9]+)")
  nouts <- gsub(nouts, "\\1", cnames, perl = TRUE)
  uouts <- unique(nouts)
  nouts <- length(uouts)

  # Build the stan data list
  stan_data <- list(
    N = nsubs,
    T = ntasks,
    ndat = nrow(dat),
    choice = dat$choice
  )

  # Right now, forcing the 2 option, 3 outcome lotteries
  for (val in c("prob", "out")) {
    for (opt in 1:2) {
      for (out in 1:4) {
        cname <- paste0("opt", opt, "_", val, out)
        stan_data[[cname]] <- dat[[cname]]
      }
    }
  }

  # We add the max and minimum outcomes for Contextual Utility
  stan_data[["Max"]] <- dat[["Max"]]
  stan_data[["Min"]] <- dat[["Min"]]


  stan_opts$file <- fname
  stan_opts$data <- stan_data

  # The fitted model
  fit <- do.call(rstan::stan, stan_opts)

  fit
}

#' Flatten the fitted model
#' @export
flatten_fit <- function(fit) {
  # Flatten it to a single matrix
  d <- dim(rstan::extract(fit, permuted = FALSE))
  niter   <- d[1]
  nchains <- d[2]

  fit <- as.matrix(fit)
  fit <- as.data.frame(fit)
  # Change the naming convention for the array parameters
  cnames <- colnames(fit)
  cnames <- gsub("]$", "", cnames)
  cnames <- gsub("\\[([0-9]+)$", "_\\1", cnames)
  colnames(fit) <- cnames

  fit$iter  <- rep(seq_len(niter))
  fit$chain <- rep(seq_len(nchains), each = niter)

  fit
}

#' Fit a model, flatten it, and write to dta
#' @export
fit_to_dta <- function(infile, outfile = "post.dta", mod_num = 1, stan_opts = list()) {

  # Read in the stata dataset
  dat <- as.data.frame(haven::read_dta(infile))

  # Fit the Stan model
  fit <- run_stan(dat, mod_num = mod_num, stan_opts = stan_opts)

  # Flatten it to a single matrix
  fit <- flatten_fit(fit)

  # If given an outfile write it, otherwise return the flattened object
  if (outfile != "") {
    haven::write_dta(fit, path = outfile)
  } else {
    return(fit)
  }
}
