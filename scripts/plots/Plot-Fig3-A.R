# Fig 3A: boxplot comparing prediction scores across feature combinations
# (FCMap / Structs / FCMap+Structs) for each contrast, with paired t-tests.
library(ggpubr)
library(tidyverse)
library(rstatix)

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

save_dir <- file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig3")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# --- Load and rename contrasts/features ---
data <- read.table(file.path(DATA_DIR, "Fig3", "Diff_Featues_Corr.csv"),
                   header = TRUE, sep = ",")

con_map <- c(
  LW_LPAC_MeanSM     = "Left SpPerAct",
  LW_LPAC_Story_Math = "Left SpComAct",
  RW_RPAC_MeanSM     = "Right SpPerAct",
  RW_RPAC_Story_Math = "Right SpComAct"
)
for (k in names(con_map)) data$Con[data$Con == k] <- con_map[[k]]
data$features[data$features == "FCMap_Structs"] <- "FCMap + Structs"

df <- data %>%
  mutate(Features = as.factor(features), Con = as.factor(Con)) %>%
  select(-features)

# --- Plot ---
colors <- c("#2E3440", "#5E81AC", "#88C0D0")
bxp <- ggplot(df, aes(x = Con, y = MeanCorr, color = Features, fill = Features)) +
  geom_boxplot(outlier.shape = NA, alpha = 1, show.legend = TRUE) +
  scale_fill_manual("Features", values = colors) +
  scale_colour_manual("Features", values = c("black", "black", "black")) +
  theme_pubr(base_size = 20, x.text.angle = 0) +
  theme(
    legend.position = "top",
    plot.title      = element_text(size = 20, hjust = 0.5),
    axis.title.y    = element_text(size = 30, hjust = 0.5),
    axis.title.x    = element_text(size = 15, hjust = 0.5, vjust = 1),
    legend.title    = element_blank(),
    legend.text     = element_text(size = 20)
  ) +
  xlab("") + ylab("Prediction Score") +
  scale_x_discrete(limits = c("Left SpPerAct", "Right SpPerAct",
                              "Left SpComAct", "Right SpComAct"))

# --- Paired t-test per contrast (Bonferroni-adjusted) ---
stat.test <- df %>%
  group_by(Con) %>%
  t_test(MeanCorr ~ Features, paired = TRUE, p.adjust.method = "bonferroni") %>%
  add_xy_position(x = "Con", dodge = 0.8, step.increase = 0.2)
print(stat.test)

bxp +
  stat_pvalue_manual(
    stat.test, step.group.by = "Con", tip.length = 0.0,
    bracket.nudge.y = 0.1, inherit.aes = FALSE,
    y.position = c(0.8, 0.85, 0.9)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

# --- Save ---
ggsave(file.path(save_dir, "Diff_Features_Corr_RidgeStyle.pdf"),
       width = 30, height = 20, device = "pdf",
       units = "cm", bg = "white", dpi = 1000)
