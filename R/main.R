supported_mods <- c( "eut_bhm", "rdu_power_bhm", "rdu_rw_bhm")
mod_map <- function(i) supported_mods[i]

covar_split <- function(s) {
  m <- unlist(strsplit(s, ")"))
  v <- lapply(m, function(m) {

    m <- stringr::str_remove(m, "\\(")
    m <- unlist(stringr::str_split(m, ","))
    m <- stringr::str_squish(m)
    m

  })
}

covar_unique <- function(cvars) {
  uvars <- unique(unlist(cvars))
  uvars[uvars!= ""]
}

covar_data <- function(cvars, dat) {
  uvars <- covar_unique(cvars)
  uvars <- stringr::str_sort(uvars)
  dat <- lapply(split(dat, dat$ID), function(sdat) sdat[1, ])
  dat <- do.call(rbind, dat)
  dat[, uvars]
}

covar_ucovars <- function(cvars) {
  uvars <- covar_unique(cvars)
  uvars <- stringr::str_sort(uvars)
  j <- lapply(cvars, function(v) {
    ifelse(uvars %in% v, 1, 0)
  })
  do.call(cbind, j)
}

covar_nest <- function(cvars) {
  count <- 0
  for (i in cvars) {
    j <- i[i != ""]
    count <- count + length(j)
  }
  count
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

#' Get the packaged file locally
#' @export
get_files <- function() {
  # Get the file for the stan code
  for (i in seq_along(supported_mods)) {
    fname <- get_stan_code(i)
    file.copy(from = fname, to = getwd(), overwrite = TRUE)
  }
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
run_stan <- function(dat, covars, fname, stan_opts = list()) {

  # Make sure the dat passes the checks
  check_dat(dat)

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

  # Get the covariates
  csplit     <- covar_split(covars)
  nhyper     <- length(csplit)
  ncvars     <- length(covar_unique(csplit))
  ncovar_est <- covar_nest(csplit)
  cvarmap    <- covar_ucovars(csplit)
  cdat       <- covar_data(csplit, dat)

  # Uncomment to help with debugging
  #print(cvarmap)
  #print(nhyper)
  #print(ncvars)
  #print(ncovar_est)

  stan_data$ncovar_est <- ncovar_est
  stan_data$ncvars     <- ncvars
  stan_data$nhyper     <- nhyper
  stan_data$cvarmap    <- cvarmap
  stan_data$covars     <- cdat

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
fit_to_dta <- function(infile, outfile = "post.dta",
                       stan_file = NA,
                       covars = NA,
                       stan_opts = list()) {

  # Read in the stata dataset
  dat <- as.data.frame(haven::read_dta(infile))

  # Write the supported models to the current dir
  get_files()

  # Get the file for the stan code
  if (is.na(stan_file)) {
    msg <- paste0("You need to specify a stan file")
    stop(msg)
  } else if (! file.exists(stan_file)) {
    msg <- paste0( "The file '", stan_file, "' does not exist")
    stop(msg)
  }

  # Fit the Stan model
  fit <- run_stan(dat, covars = covars, fname = stan_file, stan_opts = stan_opts)
  # Save the fitted model
  save(fit, file = paste0(stan_file, ".Rda"))

  # Flatten it to a single matrix
  fit <- flatten_fit(fit)

  # If given an outfile write it, otherwise return the flattened object
  if (outfile != "") {
    haven::write_dta(fit, path = outfile)
  } else {
    return(fit)
  }
}
