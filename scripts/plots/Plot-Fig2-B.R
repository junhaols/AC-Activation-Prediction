# Fig 2B: raincloud plot of per-subject prediction scores across 4 contrasts
# (HCP dataset, paired t-tests with Bonferroni correction).
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

save_dir <- file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig2")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# --- Load and reshape ---
data_path <- file.path(PROJECT_ROOT, "raw", "stat_data", "DataForStats766.csv")
data <- read.csv(data_path)

df <- data %>%
  select(LPAC_MeanSM_Corr, RPAC_MeanSM_Corr, LPAC_Story_Math_Corr, RPAC_Story_Math_Corr) %>%
  rename(
    "Left SpPerAct"  = LPAC_MeanSM_Corr,
    "Right SpPerAct" = RPAC_MeanSM_Corr,
    "Left SpComAct"  = LPAC_Story_Math_Corr,
    "Right SpComAct" = RPAC_Story_Math_Corr
  ) %>%
  pivot_longer(cols = everything(), names_to = "Contrasts", values_to = "PS") %>%
  mutate(Contrasts = factor(Contrasts,
    levels = c("Left SpPerAct", "Right SpPerAct", "Left SpComAct", "Right SpComAct"))) %>%
  mutate(Contrasts_label = factor(
    str_replace_all(Contrasts, " ", "\n"),
    levels = str_replace_all(levels(Contrasts), " ", "\n")))

# --- Plot ---
my_comparisons <- list(
  c("Left\nSpPerAct",  "Right\nSpPerAct"),
  c("Left\nSpComAct",  "Right\nSpComAct"),
  c("Left\nSpPerAct",  "Left\nSpComAct"),
  c("Right\nSpPerAct", "Right\nSpComAct")
)

p <- ggplot(df, aes(x = Contrasts_label, y = PS, fill = Contrasts)) +
  stat_dots(side = "left", justification = 1.1, binwidth = 0.008,
            dotsize = 1.2, alpha = 0.5,
            aes(color = Contrasts), show.legend = FALSE) +
  geom_boxplot(width = 0.08, outlier.shape = NA, alpha = 0.6,
               position = position_dodge(width = 0),
               aes(color = Contrasts), show.legend = FALSE) +
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0,
               justification = -0.2, point_colour = NA,
               aes(color = Contrasts), show.legend = FALSE) +
  stat_compare_means(
    comparisons = my_comparisons, method = "t.test",
    paired = TRUE, p.adjust.method = "bonferroni",
    label = "p.signif", tip.length = 0.0, bracket.size = 0.4,
    step.increase = 0.09,
    symnum.args = list(
      cutpoints = c(0, 0.001, 0.01, 0.05, 1),
      symbols   = c("***", "**", "*", "ns")
    )
  ) +
  scale_fill_npg() + scale_color_npg() +
  scale_x_discrete(expand = expansion(mult = c(0.05, 0.05))) +
  labs(title = "HCP Dataset", x = NULL, y = "Prediction Score") +
  coord_cartesian(ylim = c(-1, 1.3), xlim = c(0.6, NA)) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray60", size = 0.3) +
  theme_classic(base_size = 16) +
  theme(
    plot.title         = element_text(hjust = 0.5, size = rel(1.2)),
    axis.text.x        = element_text(size = rel(0.9), color = "black"),
    axis.text.y        = element_text(size = rel(0.9), color = "black"),
    axis.title.y       = element_text(size = rel(1.1)),
    axis.line          = element_line(linewidth = 0.5),
    axis.ticks         = element_line(linewidth = 0.5),
    legend.position    = "none",
    plot.margin        = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )

# --- Save (PDF + PNG) ---
out_base <- "hcp_Language_contrasts_corr_4_comparisons_aligned"
common <- list(plot = p, width = 18, height = 12, units = "cm",
               bg = "white", dpi = 1000)
do.call(ggsave, c(list(file.path(save_dir, paste0(out_base, ".pdf"))), common))
do.call(ggsave, c(list(file.path(save_dir, paste0(out_base, ".png"))), common))
