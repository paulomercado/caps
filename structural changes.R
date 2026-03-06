library(strucchange)
library(zoo)
library(dplyr)

# Read the CSV
# Read data
df <- read.csv("D:/School/ADMU/M AMF/caps_git/Data/cordata.csv", stringsAsFactors = FALSE)

# Convert date
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")

# Remove broken 'date' column
df$date <- NULL

# Keep only relevant columns
keep_cols <- c("Date", "Tax.Revenues", "BIR", "BOC", "Expenditures")
df_sel <- df[, keep_cols]
df_sel_92 <- df_sel %>% filter(Date >= as.Date("1992-01-01"))
# Rename for convenience
names(df_sel_92) <- c("Date", "Tax", "BIR", "BOC", "Exp")

# Check structure
str(df_sel)
df_growth <- df_sel_92 %>%
  arrange(Date) %>%
  mutate(across(-Date, 
                ~ (. / dplyr::lag(., 12) - 1) * 100, 
                .names = "{.col}_growth"))   # removes first 12 rows with NA

# Check
names(df_growth)
df_growth <- df_growth %>%
  filter(if_all(ends_with("_growth"), ~ !is.na(.)))


tax_ts <- zoo(df_growth$Tax_growth, order.by = df_growth$Date)
bir_ts <- zoo(df_growth$BIR_growth, order.by = df_growth$Date)
boc_ts <- zoo(df_growth$BOC_growth, order.by = df_growth$Date)
exp_ts <- zoo(df_growth$Exp_growth, order.by = df_growth$Date)

h_value <- 12   # or whatever you used
bp_tax   <- breakpoints(tax_ts ~ 1, h = h_value)
bp_bir   <- breakpoints(bir_ts ~ 1, h = h_value)
bp_boc   <- breakpoints(boc_ts ~ 1, h = h_value)
bp_exp   <- breakpoints(exp_ts ~ 1, h = h_value)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1) + 0.1, oma = c(0, 0, 2, 0))

# --- Tax Revenues – 6 breaks ---
n_tax <- 6
bp_idx_tax <- breakpoints(bp_tax, breaks = n_tax)$breakpoints
plot(tax_ts, main = paste0("Tax Revenues – ", n_tax, " Breaks"),
     xlab = "Date", ylab = "YoY Growth (%)", col = "darkgray", lwd = 1, type = "l")
lines(fitted(bp_tax, breaks = n_tax), col = "blue", lwd = 2)
abline(v = index(tax_ts)[bp_idx_tax], col = "red", lty = 2, lwd = 1.5)
ci_tax <- confint(bp_tax, breaks = n_tax)
for (i in 1:n_tax) {
  rect(index(tax_ts)[ci_tax$confint[i, 1]], par("usr")[3],
       index(tax_ts)[ci_tax$confint[i, 3]], par("usr")[4],
       col = rgb(1, 0, 0, 0.1), border = NA)
}
legend("bottomleft", legend = c("Observed", "Regime mean", "Break"),
       col = c("darkgray", "blue", "red"), lty = c(1, 1, 2), bty = "n")

# --- BIR – 1 break ---
n_bir <- 1
bp_idx_bir <- breakpoints(bp_bir, breaks = n_bir)$breakpoints
plot(bir_ts, main = paste0("BIR – ", n_bir, " Break"),
     xlab = "Date", ylab = "YoY Growth (%)", col = "darkgray", lwd = 1, type = "l")
lines(fitted(bp_bir, breaks = n_bir), col = "blue", lwd = 2)
abline(v = index(bir_ts)[bp_idx_bir], col = "red", lty = 2, lwd = 1.5)
ci_bir <- confint(bp_bir, breaks = n_bir)
for (i in 1:n_bir) {
  rect(index(bir_ts)[ci_bir$confint[i, 1]], par("usr")[3],
       index(bir_ts)[ci_bir$confint[i, 3]], par("usr")[4],
       col = rgb(1, 0, 0, 0.1), border = NA)
}
legend("bottomleft", legend = c("Observed", "Regime mean", "Break"),
       col = c("darkgray", "blue", "red"), lty = c(1, 1, 2), bty = "n")

# --- BOC – 9 breaks ---
n_boc <-9
bp_idx_boc <- breakpoints(bp_boc, breaks = n_boc)$breakpoints
plot(boc_ts, main = paste0("BOC – ", n_boc, " Breaks"),
     xlab = "Date", ylab = "YoY Growth (%)", col = "darkgray", lwd = 1, type = "l")
lines(fitted(bp_boc, breaks = n_boc), col = "blue", lwd = 2)
abline(v = index(boc_ts)[bp_idx_boc], col = "red", lty = 2, lwd = 1.5)
ci_boc <- confint(bp_boc, breaks = n_boc)
for (i in 1:n_boc) {
  rect(index(boc_ts)[ci_boc$confint[i, 1]], par("usr")[3],
       index(boc_ts)[ci_boc$confint[i, 3]], par("usr")[4],
       col = rgb(1, 0, 0, 0.1), border = NA)
}
legend("bottomleft", legend = c("Observed", "Regime mean", "Break"),
       col = c("darkgray", "blue", "red"), lty = c(1, 1, 2), bty = "n")

# --- Title ---
frame()
mtext("Bai-Perron Structural Breaks (h = 12)",
      outer = TRUE, cex = 1.2, line = 0.5)
par(mfrow = c(1, 1))

# --- Summary stats ---
for (info in list(
  list(name = "Tax Revenues", ts = tax_ts, bp = bp_tax, n = n_tax),
  list(name = "BIR",          ts = bir_ts, bp = bp_bir, n = n_bir),
  list(name = "BOC",          ts = boc_ts, bp = bp_boc, n = n_boc)
)) {
  cat("\n===", info$name, "===\n")
  cat("Break dates:\n")
  print(index(info$ts)[breakpoints(info$bp, breaks = info$n)$breakpoints])
  cat("Coefficients:\n")
  print(coef(info$bp, breaks = info$n))
  cat("Segment stats:\n")
  seg <- breakfactor(info$bp, breaks = info$n)
  print(tapply(coredata(info$ts), seg, function(x) c(mean=mean(x), sd=sd(x), n=length(x))))
}

