# Fig 5: scatter plots of brain measures vs. prediction score
# (LPAC/RPAC x tStd/tPeak, plus PicVocab and g-factor for LPAC).
# The companion script Plot-Fig5-circle.R produces the circular bar overview.
library(ggpubr)
library(tidyverse)
library(geomtextpath)

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

save_dir <- paste0(file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig5"), "/")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

my_colors <- list("#1a80bb", "#8cc5e3")

# --- Reusable scatter (linear fit + correlation) ---
# Always reads df columns x and adjusted_y produced by the upstream Python pipeline.
scatter_plot <- function(df, key_flag, color_point, color_line, fill_se,
                         stat_y_position, fig_title, x_lab, y_lab,
                         save_path, save_width, save_height) {
  p <- ggplot(df, aes(x = x, y = adjusted_y)) +
    geom_point(size = 1, alpha = 0.5, color = color_point) +
    geom_smooth(method = lm, color = color_line, fill = fill_se, se = TRUE) +
    stat_cor(label.y = stat_y_position, r.accuracy = 0.01, size = 5) +
    theme_pubr(base_size = 15) +
    theme(
      plot.title   = element_text(size = 19, hjust = 0.5),
      axis.title.y = element_text(size = 18, hjust = 0.5),
      axis.title.x = element_text(size = 18, hjust = 0.5, vjust = 1)
    ) +
    xlab(x_lab) + ylab(y_lab) + ggtitle(fig_title)

  if (key_flag == "tPeak") {
    p <- p +
      scale_x_continuous(breaks = seq(-2.5, 12.5, 2.5)) +
      scale_y_continuous(breaks = seq(-0.5, 1.25, 0.25), limits = c(-0.5, 1.25))
  } else if (key_flag == "tStd") {
    p <- p +
      scale_y_continuous(breaks = seq(-0.5, 1.25, 0.25), limits = c(-0.5, 1.25))
  }

  ggsave(save_path, width = save_width, height = save_height,
         units = "cm", bg = "white", dpi = 1000)
  p
}

# --- 1. Brain measure scatters: LPAC/RPAC x tStd/tPeak ---
file_dir <- paste0(file.path(DATA_DIR, "Fig5"), "/")
for (hemi in c("LPAC", "RPAC")) {
  for (key_flag in c("tStd", "tPeak")) {
    file <- paste0(file_dir, hemi, "_MeanSM_", key_flag, "_GLM.csv")
    save_path <- paste0(save_dir, hemi, "_", key_flag, "_PredCorr.pdf")
    df <- read.table(file, header = TRUE, sep = ",")

    color_idx <- if (hemi == "LPAC") 1 else 2
    fig_title <- if (hemi == "LPAC") "Left AC" else "Right AC"
    x_lab     <- if (key_flag == "tStd") "SpPerActSD" else "SpPerActPeak"

    scatter_plot(
      df, key_flag = key_flag,
      color_point = my_colors[color_idx],
      color_line  = my_colors[color_idx],
      fill_se     = my_colors[color_idx],
      stat_y_position = 1.2,
      fig_title = fig_title,
      x_lab = x_lab, y_lab = "PS of SpPerAct",
      save_path = save_path,
      save_width = 12, save_height = 12
    )
  }
}

# --- 2. PicVocab vs LPAC prediction score ---
hemi <- "LPAC"; key_flag <- "PicVocab_AgeAdj"
file <- paste0(file_dir, key_flag, "_", hemi, "_GLM.csv")
save_path <- paste0(save_dir, hemi, "_", key_flag, "_PredCorr.pdf")
df <- read.table(file, header = TRUE, sep = ",")

scatter_plot(
  df, key_flag = key_flag,
  color_point = my_colors[1], color_line = my_colors[1], fill_se = my_colors[1],
  stat_y_position = 32,
  fig_title = "Left AC",
  x_lab = "PS of SpPerAct", y_lab = "ReadEng",
  save_path = save_path,
  save_width = 12, save_height = 12
)

# --- 3. g-factor vs LPAC prediction score ---
hemi <- "LPAC"; key_flag <- "g"
file <- paste0(file_dir, key_flag, "_", hemi, "_GLM.csv")
save_path <- paste0(save_dir, hemi, "_", key_flag, "_PredCorr.pdf")
df <- read.table(file, header = TRUE, sep = ",")

scatter_plot(
  df, key_flag = key_flag,
  color_point = my_colors[1], color_line = my_colors[1], fill_se = my_colors[1],
  stat_y_position = 4.3,
  fig_title = "Left AC",
  x_lab = "PS of SpPerAct", y_lab = "g",
  save_path = save_path,
  save_width = 12, save_height = 12
)
