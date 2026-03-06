library(strucchange)
library(zoo)
library(dplyr)

# ── Load & Prep ──────────────────────────────────────────
df <- read.csv('/Users/pjam/Desktop/School/M AMF/caps/Data/cordata.csv', stringsAsFactors = FALSE)
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df$date <- NULL

keep_cols <- c("Date", "Tax.Revenues", "BIR", "BOC", "Expenditures")
df_sel <- df[, keep_cols]
df_sel_92 <- df_sel %>% filter(Date >= as.Date("1992-01-01"))
names(df_sel_92) <- c("Date", "Tax", "BIR", "BOC", "Exp")

# ── YoY Growth ───────────────────────────────────────────
df_growth <- df_sel_92 %>%
  arrange(Date) %>%
  mutate(across(-Date, ~ (. / dplyr::lag(., 12) - 1) * 100, .names = "{.col}_growth")) %>%
  filter(if_all(ends_with("_growth"), ~ !is.na(.)))

# ── Zoo Series ───────────────────────────────────────────
tax_ts <- zoo(df_growth$Tax_growth, order.by = df_growth$Date)
bir_ts <- zoo(df_growth$BIR_growth, order.by = df_growth$Date)
boc_ts <- zoo(df_growth$BOC_growth, order.by = df_growth$Date)
exp_ts <- zoo(df_growth$Exp_growth, order.by = df_growth$Date)

tax_lev <- zoo(df_sel_92$Tax, order.by = df_sel_92$Date)
bir_lev <- zoo(df_sel_92$BIR, order.by = df_sel_92$Date)
boc_lev <- zoo(df_sel_92$BOC, order.by = df_sel_92$Date)
exp_lev <- zoo(df_sel_92$Exp, order.by = df_sel_92$Date)

h_value <- 12

# ── Info lists ───────────────────────────────────────────
bp_tax <- breakpoints(tax_ts ~ 1, h = h_value)
bp_bir <- breakpoints(bir_ts ~ 1, h = h_value)
bp_boc <- breakpoints(boc_ts ~ 1, h = h_value)
bp_exp <- breakpoints(exp_ts ~ 1, h = h_value)

growth_info <- list(
  list(name = "Tax Revenues",  ts = tax_ts, bp = bp_tax, n = 6),
  list(name = "BIR",           ts = bir_ts, bp = bp_bir, n = 1),
  list(name = "BOC",           ts = boc_ts, bp = bp_boc, n = 9),
  list(name = "Expenditures",  ts = exp_ts, bp = bp_exp, n = 0)
)

level_info <- list(
  list(name = "Tax Revenues",  ts = tax_lev),
  list(name = "BIR",           ts = bir_lev),
  list(name = "BOC",           ts = boc_lev),
  list(name = "Expenditures",  ts = exp_lev)
)

# ══════════════════════════════════════════════════════════
# YoY GROWTH RATE BREAKS
# ══════════════════════════════════════════════════════════
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1) + 0.1, oma = c(0, 0, 2, 0))
for (info in growth_info) {
  n <- info$n
  plot(info$ts, main = paste0(info$name, " – ", n, " Break(s)"),
       xlab = "Date", ylab = "YoY Growth (%)", col = "darkgray", lwd = 1, type = "l")
  lines(fitted(info$bp, breaks = n), col = "blue", lwd = 2)
  
  if (n > 0) {
    bp_idx <- breakpoints(info$bp, breaks = n)$breakpoints
    abline(v = index(info$ts)[bp_idx], col = "red", lty = 2, lwd = 1.5)
    ci <- confint(info$bp, breaks = n)
    for (i in 1:n) {
      rect(index(info$ts)[ci$confint[i, 1]], par("usr")[3],
           index(info$ts)[ci$confint[i, 3]], par("usr")[4],
           col = rgb(1, 0, 0, 0.1), border = NA)
    }
  }
  
  legend("bottomleft", legend = c("Observed", "Regime mean", "Break"),
         col = c("darkgray", "blue", "red"), lty = c(1, 1, 2), bty = "n", cex = 0.8)
}
mtext("Bai-Perron — YoY Growth (h = 12)", outer = TRUE, cex = 1.2, line = 0.5)
par(mfrow = c(1, 1))

cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
cat("  GROWTH RATE BREAKS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
for (info in growth_info) {
  cat("\n===", info$name, "===\n")
  if (info$n == 0) {
    cat("No breaks detected.\n")
    cat("Overall stats:\n")
    print(c(mean = mean(coredata(info$ts)), sd = sd(coredata(info$ts)), n = length(info$ts)))
    next
  }
  cat("Break dates:\n")
  print(index(info$ts)[breakpoints(info$bp, breaks = info$n)$breakpoints])
  cat("Segment stats:\n")
  seg <- breakfactor(info$bp, breaks = info$n)
  print(tapply(coredata(info$ts), seg, function(x) c(mean = mean(x), sd = sd(x), n = length(x))))
}

# ══════════════════════════════════════════════════════════
# VARIANCE BREAKS ON GROWTH RATES
# ══════════════════════════════════════════════════════════
cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
cat("  VARIANCE BREAKS (GROWTH RATES)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (info in growth_info) {
  cat("\n===", info$name, "===\n")
  resids <- residuals(lm(coredata(info$ts) ~ 1))
  var_ts <- zoo(resids^2, order.by = index(info$ts))
  bp_var <- breakpoints(var_ts ~ 1, h = h_value)
  
  opt_bp <- breakpoints(bp_var)$breakpoints
  opt_m <- length(opt_bp)
  cat("BIC optimal m =", opt_m, "\n")
  
  if (opt_m > 0 && !any(is.na(opt_bp))) {
    cat("Break dates:\n")
    print(index(info$ts)[opt_bp])
    cat("Segment stats:\n")
    seg <- breakfactor(bp_var)
    print(tapply(coredata(var_ts), seg, function(x) c(mean_var = mean(x), sd = sd(x), n = length(x))))
  } else {
    cat("No variance breaks detected.\n")
  }
}

# ══════════════════════════════════════════════════════════
# STRUCTURAL BREAKS IN LEVELS (WITH TREND)
# ══════════════════════════════════════════════════════════
cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
cat("  STRUCTURAL BREAKS (PRICE LEVELS WITH TREND)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (info in level_info) {
  cat("\n===", info$name, "===\n")
  t_index <- seq_along(info$ts)
  bp_lev <- breakpoints(coredata(info$ts) ~ t_index, h = h_value)
  
  opt_bp <- breakpoints(bp_lev)$breakpoints
  opt_m <- length(opt_bp)
  cat("BIC optimal m =", opt_m, "\n")
  
  if (opt_m > 0 && !any(is.na(opt_bp))) {
    cat("Break dates:\n")
    print(index(info$ts)[opt_bp])
    cat("Segment stats (intercept + trend per segment):\n")
    print(coef(bp_lev))
  } else {
    cat("No breaks detected.\n")
  }
}


### testing
# ── Define segments ────────────────────────────────────────
bir_s0 <- window(bir_lev, end   = as.Date("2011-01-01"))
bir_s1 <- window(bir_lev, start = as.Date("2011-02-01"), end = as.Date("2019-12-01"))
bir_s2 <- window(bir_lev, start = as.Date("2020-01-01"))

# ── Fit trends ────────────────────────────────────────────
fit0 <- lm(coredata(bir_s0) ~ seq_along(bir_s0))
fit1 <- lm(coredata(bir_s1) ~ seq_along(bir_s1))
fit2 <- lm(coredata(bir_s2) ~ seq_along(bir_s2))

# ── Plot ──────────────────────────────────────────────────
plot(bir_lev, col = "darkgray", lwd = 1,
     main = "BIR Collections — Segment Trends",
     ylab = "BIR Collections", xlab = "Date")

lines(zoo(fitted(fit0), order.by = index(bir_s0)), col = "blue",      lwd = 2)
lines(zoo(fitted(fit1), order.by = index(bir_s1)), col = "darkorange", lwd = 2)
lines(zoo(fitted(fit2), order.by = index(bir_s2)), col = "green4",    lwd = 2)

abline(v = as.Date("2011-02-01"), col = "darkorange", lty = 2, lwd = 1.5)
abline(v = as.Date("2020-01-01"), col = "green4",     lty = 2, lwd = 1.5)

legend("topleft",
       legend = c("Observed",
                  "pre-2011 trend", "2011–2020 trend", "2020– trend",
                  "2011 break", "2020 break"),
       col    = c("darkgray", "blue", "darkorange", "green4",
                  "darkorange", "green4"),
       lty    = c(1, 1, 1, 1, 2, 2),
       lwd    = c(1, 2, 2, 2, 1.5, 1.5),
       bty    = "n", cex = 0.8)

# ── Define growth-rate segments ────────────────────────────
bir_g0 <- window(bir_ts, end   = as.Date("1997-12-01"))
bir_g1 <- window(bir_ts, start = as.Date("1998-01-01"), end = as.Date("2019-12-01"))
bir_g2 <- window(bir_ts, start = as.Date("2020-01-01"), end = as.Date("2021-12-01"))
bir_g3 <- window(bir_ts, start = as.Date("2022-01-01"))
# ── Segment means ─────────────────────────────────────────
m0 <- mean(coredata(bir_g0))
m1 <- mean(coredata(bir_g1))
m2 <- mean(coredata(bir_g2))
m3 <- mean(coredata(bir_g3))

# ── Plot ──────────────────────────────────────────────────
plot(bir_ts,
     col = "darkgray", lwd = 1, type = "l",
     main = "BIR YoY Growth — Segment Means",
     ylab = "YoY Growth (%)", xlab = "Date")

lines(zoo(rep(m0, length(bir_g0)), order.by = index(bir_g0)), col = "purple", lwd = 2)
lines(zoo(rep(m1, length(bir_g1)), order.by = index(bir_g1)), col = "blue",   lwd = 2)
lines(zoo(rep(m2, length(bir_g2)), order.by = index(bir_g2)), col = "red",    lwd = 2)
lines(zoo(rep(m3, length(bir_g3)), order.by = index(bir_g3)), col = "green4", lwd = 2)

abline(v = as.Date("1998-01-01"), col = "purple", lty = 2, lwd = 1.5)
abline(v = as.Date("2020-01-01"), col = "red",    lty = 2, lwd = 1.5)
abline(v = as.Date("2022-01-01"), col = "green4", lty = 2, lwd = 1.5)
abline(h = 0, col = "black", lty = 3, lwd = 1)

legend("bottomleft",
       legend = c("YoY Growth",
                  paste0("pre-1998 mean: ",  round(m0, 1), "%"),
                  paste0("1998–2020 mean: ", round(m1, 1), "%"),
                  paste0("2020–2022 mean: ", round(m2, 1), "%"),
                  paste0("2022– mean: ",     round(m3, 1), "%"),
                  "1998 break", "2020 break", "2022 break"),
       col    = c("darkgray", "purple", "blue", "red", "green4",
                  "purple", "red", "green4"),
       lty    = c(1, 1, 1, 1, 1, 2, 2, 2),
       lwd    = c(1, 2, 2, 2, 2, 1.5, 1.5, 1.5),
       bty    = "n", cex = 0.8)


###

# ── Define segments (BOC level trend breaks) ──────────────
boc_s0 <- window(boc_lev, end   = as.Date("2004-04-01"))
boc_s1 <- window(boc_lev, start = as.Date("2004-05-01"), end = as.Date("2008-10-01"))
boc_s2 <- window(boc_lev, start = as.Date("2008-11-01"), end = as.Date("2017-07-01"))
boc_s3 <- window(boc_lev, start = as.Date("2017-08-01"), end = as.Date("2019-12-01"))
boc_s4 <- window(boc_lev, start = as.Date("2020-01-01"), end = as.Date("2022-04-01"))
boc_s5 <- window(boc_lev, start = as.Date("2022-05-01"))

# ── Fit trends ────────────────────────────────────────────
fit0 <- lm(coredata(boc_s0) ~ seq_along(boc_s0))
fit1 <- lm(coredata(boc_s1) ~ seq_along(boc_s1))
fit2 <- lm(coredata(boc_s2) ~ seq_along(boc_s2))
fit3 <- lm(coredata(boc_s3) ~ seq_along(boc_s3))
fit4 <- lm(coredata(boc_s4) ~ seq_along(boc_s4))
fit5 <- lm(coredata(boc_s5) ~ seq_along(boc_s5))

# ── Plot ──────────────────────────────────────────────────
plot(boc_lev, col = "darkgray", lwd = 1,
     main = "BOC Collections — Segment Trends",
     ylab = "BOC Collections", xlab = "Date")

lines(zoo(fitted(fit0), order.by = index(boc_s0)), col = "purple",     lwd = 2)
lines(zoo(fitted(fit1), order.by = index(boc_s1)), col = "blue",       lwd = 2)
lines(zoo(fitted(fit2), order.by = index(boc_s2)), col = "darkorange", lwd = 2)
lines(zoo(fitted(fit3), order.by = index(boc_s3)), col = "brown",      lwd = 2)
lines(zoo(fitted(fit4), order.by = index(boc_s4)), col = "red",        lwd = 2)
lines(zoo(fitted(fit5), order.by = index(boc_s5)), col = "green4",     lwd = 2)

abline(v = as.Date("2004-05-01"), col = "purple",     lty = 2, lwd = 1.5)
abline(v = as.Date("2008-11-01"), col = "blue",       lty = 2, lwd = 1.5)
abline(v = as.Date("2017-08-01"), col = "darkorange", lty = 2, lwd = 1.5)
abline(v = as.Date("2020-01-01"), col = "brown",      lty = 2, lwd = 1.5)
abline(v = as.Date("2022-05-01"), col = "red",        lty = 2, lwd = 1.5)

legend("topleft",
       legend = c("Observed",
                  "pre-2004 trend", "2004–2008 trend", "2008–2017 trend",
                  "2017–2020 trend", "2020–2022 trend", "2022– trend",
                  "2004 break", "2008 break", "2017 break",
                  "2020 break", "2022 break"),
       col    = c("darkgray", "purple", "blue", "darkorange",
                  "brown", "red", "green4",
                  "purple", "blue", "darkorange", "brown", "red"),
       lty    = c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
       lwd    = c(1, 2, 2, 2, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5),
       bty    = "n", cex = 0.7)


# ── Define segments (BOC YoY growth breaks) ───────────────
boc_g0 <- window(boc_ts, end   = as.Date("1996-10-01"))
boc_g1 <- window(boc_ts, start = as.Date("1996-11-01"), end = as.Date("1999-02-01"))
boc_g2 <- window(boc_ts, start = as.Date("1999-03-01"), end = as.Date("2005-08-01"))
boc_g3 <- window(boc_ts, start = as.Date("2005-09-01"), end = as.Date("2006-11-01"))
boc_g4 <- window(boc_ts, start = as.Date("2006-12-01"), end = as.Date("2008-11-01"))
boc_g5 <- window(boc_ts, start = as.Date("2008-12-01"), end = as.Date("2009-11-01"))
boc_g6 <- window(boc_ts, start = as.Date("2009-12-01"), end = as.Date("2020-01-01"))
boc_g7 <- window(boc_ts, start = as.Date("2020-02-01"), end = as.Date("2021-01-01"))
boc_g8 <- window(boc_ts, start = as.Date("2021-02-01"), end = as.Date("2022-12-01"))
boc_g9 <- window(boc_ts, start = as.Date("2023-01-01"))

# ── Segment means ─────────────────────────────────────────
m0 <- mean(coredata(boc_g0))
m1 <- mean(coredata(boc_g1))
m2 <- mean(coredata(boc_g2))
m3 <- mean(coredata(boc_g3))
m4 <- mean(coredata(boc_g4))
m5 <- mean(coredata(boc_g5))
m6 <- mean(coredata(boc_g6))
m7 <- mean(coredata(boc_g7))
m8 <- mean(coredata(boc_g8))
m9 <- mean(coredata(boc_g9))

# ── Plot ──────────────────────────────────────────────────
plot(boc_ts, col = "darkgray", lwd = 1, type = "l",
     main = "BOC YoY Growth — Segment Means",
     ylab = "YoY Growth (%)", xlab = "Date")

lines(zoo(rep(m0, length(boc_g0)), order.by = index(boc_g0)), col = "purple",     lwd = 2)
lines(zoo(rep(m1, length(boc_g1)), order.by = index(boc_g1)), col = "blue",       lwd = 2)
lines(zoo(rep(m2, length(boc_g2)), order.by = index(boc_g2)), col = "darkorange", lwd = 2)
lines(zoo(rep(m3, length(boc_g3)), order.by = index(boc_g3)), col = "darkgreen",  lwd = 2)
lines(zoo(rep(m4, length(boc_g4)), order.by = index(boc_g4)), col = "brown",      lwd = 2)
lines(zoo(rep(m5, length(boc_g5)), order.by = index(boc_g5)), col = "pink",       lwd = 2)
lines(zoo(rep(m6, length(boc_g6)), order.by = index(boc_g6)), col = "steelblue",  lwd = 2)
lines(zoo(rep(m7, length(boc_g7)), order.by = index(boc_g7)), col = "red",        lwd = 2)
lines(zoo(rep(m8, length(boc_g8)), order.by = index(boc_g8)), col = "green4",     lwd = 2)
lines(zoo(rep(m9, length(boc_g9)), order.by = index(boc_g9)), col = "black",      lwd = 2)

abline(v = as.Date("1996-11-01"), col = "purple",     lty = 2, lwd = 1.2)
abline(v = as.Date("1999-03-01"), col = "blue",       lty = 2, lwd = 1.2)
abline(v = as.Date("2005-09-01"), col = "darkorange", lty = 2, lwd = 1.2)
abline(v = as.Date("2006-12-01"), col = "darkgreen",  lty = 2, lwd = 1.2)
abline(v = as.Date("2008-12-01"), col = "brown",      lty = 2, lwd = 1.2)
abline(v = as.Date("2009-12-01"), col = "pink",       lty = 2, lwd = 1.2)
abline(v = as.Date("2020-02-01"), col = "red",        lty = 2, lwd = 1.2)
abline(v = as.Date("2021-02-01"), col = "green4",     lty = 2, lwd = 1.2)
abline(v = as.Date("2023-01-01"), col = "black",      lty = 2, lwd = 1.2)
abline(h = 0, col = "black", lty = 3, lwd = 1)

legend("bottomleft",
       legend = c("YoY Growth",
                  paste0("pre-1997: ",       round(m0, 1), "%"),
                  paste0("1997-1999: ",      round(m1, 1), "%"),
                  paste0("1999-2005: ",      round(m2, 1), "%"),
                  paste0("2005-2006: ",      round(m3, 1), "%"),
                  paste0("2006-2008: ",      round(m4, 1), "%"),
                  paste0("2008-2009: ",      round(m5, 1), "%"),
                  paste0("2009-2020: ",      round(m6, 1), "%"),
                  paste0("2020-2021: ",      round(m7, 1), "%"),
                  paste0("2021-2023: ",      round(m8, 1), "%"),
                  paste0("2023-: ",          round(m9, 1), "%")),
       col    = c("darkgray", "purple", "blue", "darkorange", "darkgreen",
                  "brown", "pink", "steelblue", "red", "green4", "black"),
       lty    = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
       lwd    = c(1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
       bty    = "n", cex = 0.65)
