# Fig S3: DiceAUC raincloud plot across 4 contrasts (paired t-tests for hemisphere comparison).
library(tidyverse)
library(ggpubr)
library(ggdist)
library(ggsci)

# Resolve project paths (works in Rscript, RStudio, source())
.this <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
if (is.null(.this)) {
  .args <- commandArgs(trailingOnly = FALSE)
  .m <- grep("^--file=", .args, value = TRUE)
  if (length(.m)) .this <- sub("^--file=", "", .m[1])
}
if (is.null(.this) && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
  .this <- rstudioapi::getSourceEditorContext()$path
SCRIPT_DIR   <- if (!is.null(.this) && nchar(.this) > 0) dirname(.this) else getwd()
PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
DATA_DIR     <- file.path(PROJECT_ROOT, "raw", "figs_csv")

save_dir <- paste0(file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig_S3"), "/")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# --- Data prep ---
file1 <- file.path(DATA_DIR, "Fig_S3", "LW_LPAC_MeanSM_rptsDiceAUC.csv")
file2 <- file.path(DATA_DIR, "Fig_S3", "RW_RPAC_MeanSM_rptsDiceAUC.csv")
file3 <- file.path(DATA_DIR, "Fig_S3", "LW_LPAC_Story_Math_rptsDiceAUC.csv")
file4 <- file.path(DATA_DIR, "Fig_S3", "RW_RPAC_Story_Math_rptsDiceAUC.csv")

data1 <- read.table(file1, header = TRUE, sep = ",")
data2 <- read.table(file2, header = TRUE, sep = ",")
data3 <- read.table(file3, header = TRUE, sep = ",")
data4 <- read.table(file4, header = TRUE, sep = ",")

df <- data.frame(
  "Left SpPerAct"  = data1$mean,
  "Right SpPerAct" = data2$mean,
  "Left SpComAct"  = data3$mean,
  "Right SpComAct" = data4$mean
)

df_melt <- gather(df, "Contrasts", "DiceAUC",
                  "Left.SpPerAct", "Right.SpPerAct",
                  "Left.SpComAct", "Right.SpComAct")
df_melt$Contrasts[df_melt$Contrasts == "Left.SpPerAct"]  <- "Left SpPerAct"
df_melt$Contrasts[df_melt$Contrasts == "Right.SpPerAct"] <- "Right SpPerAct"
df_melt$Contrasts[df_melt$Contrasts == "Left.SpComAct"]  <- "Left SpComAct"
df_melt$Contrasts[df_melt$Contrasts == "Right.SpComAct"] <- "Right SpComAct"
df_melt$Contrasts <- factor(df_melt$Contrasts,
  levels = c("Left SpPerAct", "Right SpPerAct", "Left SpComAct", "Right SpComAct"))

df <- df_melt %>%
  rename(PS = DiceAUC) %>%
  mutate(Contrasts_label = factor(
    str_replace_all(Contrasts, " ", "\n"),
    levels = str_replace_all(levels(Contrasts), " ", "\n")))

# --- Paired t-tests: SpPerAct vs SpComAct within each hemisphere ---
left_test  <- t.test(data1$mean, data3$mean, paired = TRUE)
right_test <- t.test(data2$mean, data4$mean, paired = TRUE)

cat("\n=== Left hemisphere (SpPerAct vs SpComAct) ===\n")
cat("mean diff:", mean(data1$mean - data3$mean),
    "  t:", left_test$statistic,
    "  p:", left_test$p.value, "\n")
cat("\n=== Right hemisphere (SpPerAct vs SpComAct) ===\n")
cat("mean diff:", mean(data2$mean - data4$mean),
    "  t:", right_test$statistic,
    "  p:", right_test$p.value, "\n")

format_pvalue <- function(p) {
  if (p < 0.001) "***"
  else if (p < 0.01) "**"
  else if (p < 0.05) "*"
  else "ns"
}
left_sig  <- format_pvalue(left_test$p.value)
right_sig <- format_pvalue(right_test$p.value)

# --- Raincloud plot ---
p <- ggplot(df, aes(x = Contrasts_label, y = PS, fill = Contrasts)) +
  stat_dots(side = "left", justification = 1.2, binwidth = NA,
            dotsize = 0.5, alpha = 0.7, scale = 0.6,
            aes(color = Contrasts), show.legend = FALSE) +
  geom_boxplot(width = 0.05, outlier.shape = NA, alpha = 0.8,
               position = position_identity(),
               aes(color = Contrasts), show.legend = FALSE) +
  stat_halfeye(adjust = 1.0, width = 0.25, .width = 0,
               justification = -0.5, point_colour = NA,
               aes(color = Contrasts), show.legend = FALSE) +
  # Significance brackets
  annotate("segment", x = 1, xend = 3, y = 0.42, yend = 0.42,
           color = "black", linewidth = 0.5) +
  annotate("text", x = 2, y = 0.43, label = left_sig,
           size = 5, color = "black") +
  annotate("segment", x = 2, xend = 4, y = 0.39, yend = 0.39,
           color = "black", linewidth = 0.5) +
  annotate("text", x = 3, y = 0.40, label = right_sig,
           size = 5, color = "black") +
  scale_fill_npg() + scale_color_npg() +
  scale_x_discrete(expand = expansion(mult = c(0.15, 0.15))) +
  labs(title = "HCP dataset", x = NULL, y = "DiceAUC") +
  coord_cartesian(ylim = c(0.0, 0.45), xlim = c(0.6, NA)) +
  geom_hline(yintercept = 0.2, linetype = "dashed",
             color = "gray60", linewidth = 0.4) +
  theme_classic(base_size = 14) +
  theme(
    plot.title       = element_text(hjust = 0.5, size = rel(1.1)),
    axis.text.x      = element_text(size = rel(0.95), color = "black",
                                    margin = margin(t = 5)),
    axis.text.y      = element_text(size = rel(0.95), color = "black"),
    axis.title.y     = element_text(size = rel(1.0)),
    axis.line        = element_line(linewidth = 0.6),
    axis.ticks       = element_line(linewidth = 0.5),
    legend.position  = "none",
    plot.margin      = unit(c(0.8, 0.8, 0.5, 0.5), "cm"),
    panel.grid       = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA)
  )

# --- Save (PDF + PNG + TIFF) ---
out_base <- "Language_DiceAUC_4_contrasts_clean"
common <- list(plot = p, width = 20, height = 10, units = "cm", bg = "white")
do.call(ggsave, c(list(file.path(save_dir, paste0(out_base, ".pdf")),  dpi = 300), common))
do.call(ggsave, c(list(file.path(save_dir, paste0(out_base, ".png")),  dpi = 300), common))
do.call(ggsave, c(list(file.path(save_dir, paste0(out_base, ".tiff")), dpi = 600,
                       compression = "lzw"), common))
