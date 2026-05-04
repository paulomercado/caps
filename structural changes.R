library(strucchange)
library(zoo)
library(dplyr)

# ── Load & Prep ──────────────────────────────────────────
df <- read.csv('/Users/pjam/Desktop/School/M AMF/caps/Data/cordata_breaks.csv',
               stringsAsFactors = FALSE)
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df$date <- NULL

df_sel    <- df[, c("Date", "Tax.Revenues", "BIR", "BOC", "Expenditures", "Non.tax.Revenues", "Outstanding.Debt")]
df_sel_92 <- df_sel %>% filter(Date >= as.Date("1992-01-01"))
names(df_sel_92) <- c("Date", "Tax", "BIR", "BOC", "Exp", "NTR", "Debt")

# Outstanding Debt starts 1993 — separate subset
df_sel_debt <- df_sel %>% filter(Date >= as.Date("1993-01-01"))
names(df_sel_debt) <- c("Date", "Tax", "BIR", "BOC", "Exp", "NTR", "Debt")
df_sel_debt <- df_sel_debt %>% filter(!is.na(Debt))

# ── CREATE OUTPUT DIRECTORY ──────────────────────────────
plot_dir <- "plots"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# ── YoY Growth ───────────────────────────────────────────
df_growth <- df_sel_92 %>%
  arrange(Date) %>%
  mutate(across(c(Tax, BIR, BOC, Exp, NTR), ~ (. / dplyr::lag(., 12) - 1) * 100,
                .names = "{.col}_growth")) %>%
  filter(if_all(ends_with("_growth"), ~ !is.na(.)))

df_growth_debt <- df_sel_debt %>%
  arrange(Date) %>%
  mutate(Debt_growth = (Debt / dplyr::lag(Debt, 12) - 1) * 100) %>%
  filter(!is.na(Debt_growth))

# ── Zoo Series ───────────────────────────────────────────
tax_ts  <- zoo(df_growth$Tax_growth, order.by = df_growth$Date)
bir_ts  <- zoo(df_growth$BIR_growth, order.by = df_growth$Date)
boc_ts  <- zoo(df_growth$BOC_growth, order.by = df_growth$Date)
exp_ts  <- zoo(df_growth$Exp_growth, order.by = df_growth$Date)
ntr_ts  <- zoo(df_growth$NTR_growth, order.by = df_growth$Date)
debt_ts <- zoo(df_growth_debt$Debt_growth, order.by = df_growth_debt$Date)

tax_lev  <- zoo(df_sel_92$Tax,  order.by = df_sel_92$Date)
bir_lev  <- zoo(df_sel_92$BIR,  order.by = df_sel_92$Date)
boc_lev  <- zoo(df_sel_92$BOC,  order.by = df_sel_92$Date)
exp_lev  <- zoo(df_sel_92$Exp,  order.by = df_sel_92$Date)
ntr_lev  <- zoo(df_sel_92$NTR,  order.by = df_sel_92$Date)
debt_lev <- zoo(df_sel_debt$Debt, order.by = df_sel_debt$Date)

# ── Helper: BIC-optimal m ────────────────────────────────
get_opt_m <- function(bp) {
  opt <- breakpoints(bp)$breakpoints
  if (length(opt) == 0 || all(is.na(opt))) 0L else length(opt)
}

# ── Helper: get break dates from breakpoints object ──────
get_bp_dates <- function(bp, ts) {
  idx <- breakpoints(bp)$breakpoints
  if (length(idx) == 0 || all(is.na(idx))) return(as.Date(character(0)))
  index(ts)[idx]
}

# ── Helper: get segment start/end dates ──────────────────
get_seg_bounds <- function(bp_dates, all_dates) {
  if (length(bp_dates) == 0) {
    starts <- all_dates[1]
    ends   <- all_dates[length(all_dates)]
  } else {
    bp_idx <- match(bp_dates, all_dates)
    starts <- c(all_dates[1], all_dates[bp_idx + 1])
    ends   <- c(all_dates[bp_idx], all_dates[length(all_dates)])
  }
  list(starts = starts, ends = ends)
}

# ── Helper: plot segment trends (lm) using calendar dates ─
plot_seg_trends <- function(ts, bp_dates, title, ylab,
                            legend_pos  = "topleft",
                            legend_pos2 = "bottomright",
                            seg_cols = c("purple", "blue", "darkorange",
                                         "brown", "red", "green4",
                                         "steelblue", "darkgreen", "pink", "black")) {
  all_dates <- index(ts)
  bounds    <- get_seg_bounds(bp_dates, all_dates)
  n_seg     <- length(bounds$starts)
  
  plot(ts, col = "darkgray", lwd = 1, type = "l",
       main = title, ylab = ylab, xlab = "Date")
  
  leg_labels <- "Observed"
  leg_cols   <- "darkgray"
  leg_lty    <- 1; leg_lwd <- 1
  used_cols  <- character(n_seg)
  
  for (s in seq_len(n_seg)) {
    seg_ts <- window(ts, start = bounds$starts[s], end = bounds$ends[s])
    col_s  <- seg_cols[(s - 1) %% length(seg_cols) + 1]
    used_cols[s] <- col_s
    fit_s  <- lm(coredata(seg_ts) ~ seq_along(seg_ts))
    lines(zoo(fitted(fit_s), order.by = index(seg_ts)), col = col_s, lwd = 2)
    lbl <- paste0(format(bounds$starts[s], "%Y-%m"), " to ",
                  format(bounds$ends[s],   "%Y-%m"))
    leg_labels <- c(leg_labels, lbl)
    leg_cols   <- c(leg_cols, col_s)
    leg_lty    <- c(leg_lty, 1); leg_lwd <- c(leg_lwd, 2)
  }
  
  if (length(bp_dates) > 0) {
    break_cols <- used_cols[-1]
    for (i in seq_along(bp_dates))
      abline(v = bp_dates[i], col = break_cols[i], lty = 2, lwd = 1.5)
    legend(legend_pos2,
           legend = paste0(format(bp_dates, "%Y"), " break"),
           col = break_cols, lty = 2, lwd = 1.5,
           bty = "n", cex = 0.7, x.intersp = 0.6, y.intersp = 0.85)
  }
  legend(legend_pos, legend = leg_labels, col = leg_cols,
         lty = leg_lty, lwd = leg_lwd,
         bty = "n", cex = 0.7, x.intersp = 0.6, y.intersp = 0.85)
}

# ── Helper: plot segment means using calendar dates ───────
plot_seg_means <- function(ts, bp_dates, title, ylab,
                           legend_pos = "bottomleft",
                           seg_cols = c("purple", "blue", "darkorange",
                                        "brown", "red", "green4",
                                        "steelblue", "darkgreen", "pink", "black")) {
  all_dates <- index(ts)
  bounds    <- get_seg_bounds(bp_dates, all_dates)
  n_seg     <- length(bounds$starts)
  
  plot(ts, col = "darkgray", lwd = 1, type = "l",
       main = title, ylab = ylab, xlab = "Date")
  abline(h = 0, col = "black", lty = 3, lwd = 1)
  
  leg_labels <- "YoY Growth"
  leg_cols   <- "darkgray"
  leg_lty    <- 1; leg_lwd <- 1
  used_cols  <- character(n_seg)
  
  for (s in seq_len(n_seg)) {
    seg_ts <- window(ts, start = bounds$starts[s], end = bounds$ends[s])
    col_s  <- seg_cols[(s - 1) %% length(seg_cols) + 1]
    used_cols[s] <- col_s
    m_s    <- mean(coredata(seg_ts))
    lines(zoo(rep(m_s, length(seg_ts)), order.by = index(seg_ts)),
          col = col_s, lwd = 2)
    lbl <- paste0(format(bounds$starts[s], "%Y-%m"), " to ",
                  format(bounds$ends[s],   "%Y-%m"),
                  " (", round(m_s, 1), "%)")
    leg_labels <- c(leg_labels, lbl)
    leg_cols   <- c(leg_cols, col_s)
    leg_lty    <- c(leg_lty, 1); leg_lwd <- c(leg_lwd, 2)
  }
  
  if (length(bp_dates) > 0) {
    break_cols <- used_cols[-1]
    for (i in seq_along(bp_dates))
      abline(v = bp_dates[i], col = break_cols[i], lty = 2, lwd = 1.2)
  }
  
  legend(legend_pos, legend = leg_labels, col = leg_cols,
         lty = leg_lty, lwd = leg_lwd,
         bty = "n", cex = 0.65, x.intersp = 0.5, y.intersp = 0.8)
}


# ══════════════════════════════════════════════════════════
# MAIN LOOP OVER h VALUES
# ══════════════════════════════════════════════════════════
for (h_value in c(12, 24, 36)) {
  
  cat("\n\n")
  cat(paste(rep("#", 70), collapse = ""), "\n")
  cat(paste0("  h = ", h_value, "\n"))
  cat(paste(rep("#", 70), collapse = ""), "\n")
  
  # ── Growth breakpoints ───────────────────────────────────
  bp_tax_g  <- breakpoints(tax_ts  ~ 1, h = h_value)
  bp_bir_g  <- breakpoints(bir_ts  ~ 1, h = h_value)
  bp_boc_g  <- breakpoints(boc_ts  ~ 1, h = h_value)
  bp_exp_g  <- breakpoints(exp_ts  ~ 1, h = h_value)
  bp_ntr_g  <- breakpoints(ntr_ts  ~ 1, h = h_value)
  bp_debt_g <- breakpoints(debt_ts ~ 1, h = h_value)
  
  growth_info <- list(
    list(name = "Tax Revenues",     ts = tax_ts,  bp = bp_tax_g,  n = get_opt_m(bp_tax_g)),
    list(name = "BIR",              ts = bir_ts,  bp = bp_bir_g,  n = get_opt_m(bp_bir_g)),
    list(name = "BOC",              ts = boc_ts,  bp = bp_boc_g,  n = get_opt_m(bp_boc_g)),
    list(name = "Expenditures",     ts = exp_ts,  bp = bp_exp_g,  n = get_opt_m(bp_exp_g)),
    list(name = "Non-tax Revenues", ts = ntr_ts,  bp = bp_ntr_g,  n = get_opt_m(bp_ntr_g)),
    list(name = "Outstanding Debt", ts = debt_ts, bp = bp_debt_g, n = get_opt_m(bp_debt_g))
  )
  
  # ── Level breakpoints ────────────────────────────────────
  bp_tax_l  <- breakpoints(coredata(tax_lev)  ~ seq_along(tax_lev),  h = h_value)
  bp_bir_l  <- breakpoints(coredata(bir_lev)  ~ seq_along(bir_lev),  h = h_value)
  bp_boc_l  <- breakpoints(coredata(boc_lev)  ~ seq_along(boc_lev),  h = h_value)
  bp_exp_l  <- breakpoints(coredata(exp_lev)  ~ seq_along(exp_lev),  h = h_value)
  bp_ntr_l  <- breakpoints(coredata(ntr_lev)  ~ seq_along(ntr_lev),  h = h_value)
  bp_debt_l <- breakpoints(coredata(debt_lev) ~ seq_along(debt_lev), h = h_value)
  
  # ── Extract break dates (calendar) ──────────────────────
  tax_g_bd  <- get_bp_dates(bp_tax_g,  tax_ts)
  bir_g_bd  <- get_bp_dates(bp_bir_g,  bir_ts)
  boc_g_bd  <- get_bp_dates(bp_boc_g,  boc_ts)
  exp_g_bd  <- get_bp_dates(bp_exp_g,  exp_ts)
  ntr_g_bd  <- get_bp_dates(bp_ntr_g,  ntr_ts)
  debt_g_bd <- get_bp_dates(bp_debt_g, debt_ts)
  
  tax_l_bd  <- get_bp_dates(bp_tax_l,  tax_lev)
  bir_l_bd  <- get_bp_dates(bp_bir_l,  bir_lev)
  boc_l_bd  <- get_bp_dates(bp_boc_l,  boc_lev)
  exp_l_bd  <- get_bp_dates(bp_exp_l,  exp_lev)
  ntr_l_bd  <- get_bp_dates(bp_ntr_l,  ntr_lev)
  debt_l_bd <- get_bp_dates(bp_debt_l, debt_lev)
  
  # ── Named lists for looping ─────────────────────────────
  all_names_6 <- c("Tax Revenues", "BIR", "BOC", "Expenditures", "Non-tax Revenues", "Outstanding Debt")
  
  ts_g_map <- list("Tax Revenues" = tax_ts, "BIR" = bir_ts,
                   "BOC" = boc_ts, "Expenditures" = exp_ts,
                   "Non-tax Revenues" = ntr_ts, "Outstanding Debt" = debt_ts)
  bd_g_map <- list("Tax Revenues" = tax_g_bd, "BIR" = bir_g_bd,
                   "BOC" = boc_g_bd, "Expenditures" = exp_g_bd,
                   "Non-tax Revenues" = ntr_g_bd, "Outstanding Debt" = debt_g_bd)
  bp_g_map <- list("Tax Revenues" = bp_tax_g, "BIR" = bp_bir_g,
                   "BOC" = bp_boc_g, "Expenditures" = bp_exp_g,
                   "Non-tax Revenues" = bp_ntr_g, "Outstanding Debt" = bp_debt_g)
  
  ts_l_map <- list("Tax Revenues" = tax_lev, "BIR" = bir_lev,
                   "BOC" = boc_lev, "Expenditures" = exp_lev,
                   "Non-tax Revenues" = ntr_lev, "Outstanding Debt" = debt_lev)
  bd_l_map <- list("Tax Revenues" = tax_l_bd, "BIR" = bir_l_bd,
                   "BOC" = boc_l_bd, "Expenditures" = exp_l_bd,
                   "Non-tax Revenues" = ntr_l_bd, "Outstanding Debt" = debt_l_bd)
  bp_l_map <- list("Tax Revenues" = bp_tax_l, "BIR" = bp_bir_l,
                   "BOC" = bp_boc_l, "Expenditures" = bp_exp_l,
                   "Non-tax Revenues" = bp_ntr_l, "Outstanding Debt" = bp_debt_l)
  
  
  # ════════════════════════════════════════════════════════
  # Individual YoY segment means (all 6)
  # ════════════════════════════════════════════════════════
  for (nm in all_names_6) {
    plot_seg_means(ts_g_map[[nm]], bd_g_map[[nm]],
                   title = paste0(nm, " - YoY Segment Means (h=", h_value, ")"),
                   ylab  = "YoY Growth (%)")
    fname <- file.path(plot_dir, paste0("h", h_value, "_02_yoy_means_",
                                        gsub(" ", "_", nm), ".png"))
    dev.copy(png, fname, width = 1000, height = 700, res = 120)
    dev.off()
    cat("  Saved:", fname, "\n")
  }
  
  # ════════════════════════════════════════════════════════
  # Console: Growth breaks
  # ════════════════════════════════════════════════════════
  cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
  cat(paste0("  GROWTH RATE BREAKS  (h = ", h_value, ")\n"))
  cat(paste(rep("=", 60), collapse = ""), "\n")
  for (info in growth_info) {
    cat("\n===", info$name, "===\n")
    if (info$n == 0) {
      cat("No breaks detected.\n")
      print(c(mean = mean(coredata(info$ts)),
              sd   = sd(coredata(info$ts)),
              n    = length(info$ts)))
      next
    }
    cat("Break dates:\n")
    print(index(info$ts)[breakpoints(info$bp, breaks = info$n)$breakpoints])
    cat("Segment stats:\n")
    seg <- breakfactor(info$bp, breaks = info$n)
    print(tapply(coredata(info$ts), seg,
                 function(x) c(mean = mean(x), sd = sd(x), n = length(x))))
  }
  
  # ════════════════════════════════════════════════════════
  # Individual level segment trends (all 6)
  # ════════════════════════════════════════════════════════
  for (nm in all_names_6) {
    plot_seg_trends(ts_l_map[[nm]], bd_l_map[[nm]],
                    title = paste0(nm, " - Segment Trends (h=", h_value, ")"),
                    ylab  = "Level")
    fname <- file.path(plot_dir, paste0("h", h_value, "_04_level_trends_",
                                        gsub(" ", "_", nm), ".png"))
    dev.copy(png, fname, width = 1000, height = 700, res = 120)
    dev.off()
    cat("  Saved:", fname, "\n")
  }
  
  # ════════════════════════════════════════════════════════
  # Console: Level breaks
  # ════════════════════════════════════════════════════════
  cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
  cat(paste0("  STRUCTURAL BREAKS (LEVELS WITH TREND)  (h = ", h_value, ")\n"))
  cat(paste(rep("=", 60), collapse = ""), "\n")
  for (nm in all_names_6) {
    cat("\n===", nm, "===\n")
    bd <- bd_l_map[[nm]]
    bp <- bp_l_map[[nm]]
    cat("BIC optimal m =", length(bd), "\n")
    if (length(bd) > 0) {
      cat("Break dates:\n"); print(bd)
      cat("Segment coefficients (intercept + trend):\n")
      print(coef(bp))
    } else {
      cat("No breaks detected.\n")
    }
  }
  
  # ════════════════════════════════════════════════════════
  # PLOT 5: Superimposed Tax & Expenditures — YoY Growth
  # ════════════════════════════════════════════════════════
  tax_col_raw  <- rgb(0.27, 0.51, 0.71, 0.45)
  exp_col_raw  <- rgb(0.70, 0.13, 0.13, 0.45)
  ntr_col_raw  <- rgb(0.0,  0.39, 0.0,  0.45)
  tax_col_bold <- "steelblue"
  exp_col_bold <- "firebrick"
  ntr_col_bold <- "green4"
  
  fname <- file.path(plot_dir, paste0("h", h_value, "_05_yoy_tax_exp_superimposed.png"))
  
  y_range <- range(c(coredata(tax_ts), coredata(exp_ts)), na.rm = TRUE)
  
  plot(tax_ts, col = tax_col_raw, lwd = 1, type = "l",
       main = paste0("Tax Revenues vs Expenditures — YoY Growth (h=", h_value, ")"),
       ylab = "YoY Growth (%)", xlab = "Date", ylim = y_range)
  lines(exp_ts, col = exp_col_raw, lwd = 1)
  abline(h = 0, col = "black", lty = 3, lwd = 1)
  
  if (length(tax_g_bd) > 0)
    for (d in tax_g_bd) abline(v = d, col = tax_col_bold, lty = 2, lwd = 1.5)
  if (length(exp_g_bd) > 0)
    for (d in exp_g_bd) abline(v = d, col = exp_col_bold, lty = 4, lwd = 1.5)
  
  bounds_tax <- get_seg_bounds(tax_g_bd, index(tax_ts))
  for (s in seq_along(bounds_tax$starts)) {
    seg <- window(tax_ts, start = bounds_tax$starts[s], end = bounds_tax$ends[s])
    lines(zoo(rep(mean(coredata(seg)), length(seg)), order.by = index(seg)),
          col = tax_col_bold, lwd = 1.2)
  }
  
  bounds_exp <- get_seg_bounds(exp_g_bd, index(exp_ts))
  for (s in seq_along(bounds_exp$starts)) {
    seg <- window(exp_ts, start = bounds_exp$starts[s], end = bounds_exp$ends[s])
    lines(zoo(rep(mean(coredata(seg)), length(seg)), order.by = index(seg)),
          col = exp_col_bold, lwd = 1.2)
  }
  
  legend("topleft",
         legend = c("Tax Revenues", "Expenditures",
                    "Tax break", "Exp break"),
         col = c(tax_col_bold, exp_col_bold,
                 tax_col_bold, exp_col_bold),
         lty = c(1, 1, 2, 4),
         lwd = c(1, 1, 1.5, 1.5),
         bty = "n", cex = 0.7)
  
  dev.copy(png, fname, width = 1200, height = 700, res = 120)
  dev.off()
  cat("  Saved:", fname, "\n")
  
  # ════════════════════════════════════════════════════════
  # PLOT 6: Superimposed Tax & Expenditures — Levels
  # ════════════════════════════════════════════════════════
  fname <- file.path(plot_dir, paste0("h", h_value, "_06_level_tax_exp_superimposed.png"))
  
  y_range_l <- range(c(coredata(tax_lev), coredata(exp_lev)), na.rm = TRUE)
  
  plot(tax_lev, col = tax_col_raw, lwd = 1, type = "l",
       main = paste0("Tax Revenues vs Expenditures — Levels (h=", h_value, ")"),
       ylab = "Level (PHP MN)", xlab = "Date", ylim = y_range_l)
  lines(exp_lev, col = exp_col_raw, lwd = 1)
  
  bounds_tl <- get_seg_bounds(tax_l_bd, index(tax_lev))
  for (s in seq_along(bounds_tl$starts)) {
    seg <- window(tax_lev, start = bounds_tl$starts[s], end = bounds_tl$ends[s])
    fit <- lm(coredata(seg) ~ seq_along(seg))
    lines(zoo(fitted(fit), order.by = index(seg)), col = tax_col_bold, lwd = 1.2)
  }
  if (length(tax_l_bd) > 0)
    for (d in tax_l_bd) abline(v = d, col = tax_col_bold, lty = 2, lwd = 1.5)
  
  bounds_el <- get_seg_bounds(exp_l_bd, index(exp_lev))
  for (s in seq_along(bounds_el$starts)) {
    seg <- window(exp_lev, start = bounds_el$starts[s], end = bounds_el$ends[s])
    fit <- lm(coredata(seg) ~ seq_along(seg))
    lines(zoo(fitted(fit), order.by = index(seg)), col = exp_col_bold, lwd = 1.2)
  }
  if (length(exp_l_bd) > 0)
    for (d in exp_l_bd) abline(v = d, col = exp_col_bold, lty = 4, lwd = 1.5)
  
  legend("topleft",
         legend = c("Tax Revenues", "Expenditures",
                    "Tax break", "Exp break",
                    "Tax trend", "Exp trend"),
         col = c(tax_col_bold, exp_col_bold,
                 tax_col_bold, exp_col_bold,
                 tax_col_bold, exp_col_bold),
         lty = c(1, 1, 2, 4, 1, 1),
         lwd = c(1, 1, 1.5, 1.5, 1.2, 1.2),
         bty = "n", cex = 0.7)
  
  dev.copy(png, fname, width = 1200, height = 700, res = 120)
  dev.off()
  cat("  Saved:", fname, "\n")
  
  # ════════════════════════════════════════════════════════
  # PLOT 7: Superimposed Tax, Exp & NTR — Levels
  # ════════════════════════════════════════════════════════
  fname <- file.path(plot_dir, paste0("h", h_value, "_07_level_tax_exp_ntr_superimposed.png"))
  
  ylim_range3 <- range(c(coredata(tax_lev), coredata(exp_lev), coredata(ntr_lev)), na.rm = TRUE)
  
  plot(tax_lev, col = tax_col_raw, lwd = 1, type = "l",
       main = paste0("Tax Revenues vs Expenditures vs Non-tax Revenues — Levels (h=", h_value, ")"),
       ylab = "Level (PHP MN)", xlab = "Date", ylim = ylim_range3)
  lines(exp_lev, col = exp_col_raw, lwd = 1)
  lines(ntr_lev, col = ntr_col_raw, lwd = 1)
  
  for (s in seq_along(bounds_tl$starts)) {
    seg <- window(tax_lev, start = bounds_tl$starts[s], end = bounds_tl$ends[s])
    fit <- lm(coredata(seg) ~ seq_along(seg))
    lines(zoo(fitted(fit), order.by = index(seg)), col = tax_col_bold, lwd = 1.5)
  }
  for (s in seq_along(bounds_el$starts)) {
    seg <- window(exp_lev, start = bounds_el$starts[s], end = bounds_el$ends[s])
    fit <- lm(coredata(seg) ~ seq_along(seg))
    lines(zoo(fitted(fit), order.by = index(seg)), col = exp_col_bold, lwd = 1.5)
  }
  bounds_nl <- get_seg_bounds(ntr_l_bd, index(ntr_lev))
  for (s in seq_along(bounds_nl$starts)) {
    seg <- window(ntr_lev, start = bounds_nl$starts[s], end = bounds_nl$ends[s])
    fit <- lm(coredata(seg) ~ seq_along(seg))
    lines(zoo(fitted(fit), order.by = index(seg)), col = ntr_col_bold, lwd = 1.5)
  }
  
  if (length(tax_l_bd) > 0) for (d in tax_l_bd) abline(v = d, col = tax_col_bold, lty = 2, lwd = 1.5)
  if (length(exp_l_bd) > 0) for (d in exp_l_bd) abline(v = d, col = exp_col_bold, lty = 4, lwd = 1.5)
  if (length(ntr_l_bd) > 0) for (d in ntr_l_bd) abline(v = d, col = ntr_col_bold, lty = 3, lwd = 1.5)
  
  legend("topleft",
         legend = c("Tax Revenues", "Expenditures", "Non-tax Revenues",
                    "Tax break", "Exp break", "NTR break",
                    "Tax trend", "Exp trend", "NTR trend"),
         col = c(tax_col_bold, exp_col_bold, ntr_col_bold,
                 tax_col_bold, exp_col_bold, ntr_col_bold,
                 tax_col_bold, exp_col_bold, ntr_col_bold),
         lty = c(1,1,1, 2,4,3, 1,1,1),
         lwd = c(1,1,1, 1.5,1.5,1.5, 1.5,1.5,1.5),
         bty = "n", cex = 0.7)
  
  dev.copy(png, fname, width = 1200, height = 700, res = 120)
  dev.off()
  cat("  Saved:", fname, "\n")
  
  # ════════════════════════════════════════════════════════
  # PLOT 8: Superimposed Tax, Exp & NTR — YoY Growth
  # ════════════════════════════════════════════════════════
  fname <- file.path(plot_dir, paste0("h", h_value, "_08_yoy_tax_exp_ntr_superimposed.png"))
  
  y_range_3g <- range(c(coredata(tax_ts), coredata(exp_ts), coredata(ntr_ts)), na.rm = TRUE)
  
  plot(tax_ts, col = tax_col_raw, lwd = 1, type = "l",
       main = paste0("Tax Revenues vs Expenditures vs Non-tax Revenues — YoY Growth (h=", h_value, ")"),
       ylab = "YoY Growth (%)", xlab = "Date", ylim = y_range_3g)
  lines(exp_ts, col = exp_col_raw, lwd = 1)
  lines(ntr_ts, col = ntr_col_raw, lwd = 1)
  abline(h = 0, col = "black", lty = 3, lwd = 1)
  
  for (s in seq_along(bounds_tax$starts)) {
    seg <- window(tax_ts, start = bounds_tax$starts[s], end = bounds_tax$ends[s])
    lines(zoo(rep(mean(coredata(seg)), length(seg)), order.by = index(seg)),
          col = tax_col_bold, lwd = 1.2)
  }
  for (s in seq_along(bounds_exp$starts)) {
    seg <- window(exp_ts, start = bounds_exp$starts[s], end = bounds_exp$ends[s])
    lines(zoo(rep(mean(coredata(seg)), length(seg)), order.by = index(seg)),
          col = exp_col_bold, lwd = 1.2)
  }
  bounds_ntr_g <- get_seg_bounds(ntr_g_bd, index(ntr_ts))
  for (s in seq_along(bounds_ntr_g$starts)) {
    seg <- window(ntr_ts, start = bounds_ntr_g$starts[s], end = bounds_ntr_g$ends[s])
    lines(zoo(rep(mean(coredata(seg)), length(seg)), order.by = index(seg)),
          col = ntr_col_bold, lwd = 1.2)
  }
  
  if (length(tax_g_bd) > 0) for (d in tax_g_bd) abline(v = d, col = tax_col_bold, lty = 2, lwd = 1.5)
  if (length(exp_g_bd) > 0) for (d in exp_g_bd) abline(v = d, col = exp_col_bold, lty = 4, lwd = 1.5)
  if (length(ntr_g_bd) > 0) for (d in ntr_g_bd) abline(v = d, col = ntr_col_bold, lty = 3, lwd = 1.5)
  
  legend("topleft",
         legend = c("Tax Revenues", "Expenditures", "Non-tax Revenues",
                    "Tax break", "Exp break", "NTR break"),
         col = c(tax_col_bold, exp_col_bold, ntr_col_bold,
                 tax_col_bold, exp_col_bold, ntr_col_bold),
         lty = c(1,1,1, 2,4,3),
         lwd = c(1,1,1, 1.5,1.5,1.5),
         bty = "n", cex = 0.7)
  
  dev.copy(png, fname, width = 1200, height = 700, res = 120)
  dev.off()
  cat("  Saved:", fname, "\n")
  
} # ── end h_value loop ────────────────────────────────────

cat("\n\nAll plots saved to:", normalizePath(plot_dir), "\n")