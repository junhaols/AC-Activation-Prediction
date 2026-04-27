# Fig S2: per-run scatter plots (actual vs. predicted activation) on validation dataset.
library(ggpubr)
library(tidyverse)
library(svglite)

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

save_dir <- paste0(file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig_S2"), "/")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

my_colors <- list("#1a80bb", "#8cc5e3")

scatter_plot <- function(df, col_x, col_y, color_point, color_line, fill_se,
                         stat_y_position, fig_title, x_lab, y_lab,
                         save_path, save_width, save_height) {
  p <- ggplot(df, aes_string(x = col_x, y = col_y)) +
    geom_point(size = 1, alpha = 0.8, color = color_point) +
    geom_smooth(method = lm, color = color_line, fill = fill_se, se = TRUE) +
    stat_cor(label.y = stat_y_position, r.accuracy = 0.01, size = 15) +
    theme_pubr(base_size = 35) +
    theme(
      plot.title   = element_text(size = 50, hjust = 0.5),
      axis.title.y = element_text(size = 40, hjust = 0.5),
      axis.title.x = element_text(size = 40, hjust = 0.5, vjust = 1)
    ) +
    xlab(x_lab) + ylab(y_lab) + ggtitle(fig_title) +
    scale_y_continuous(breaks = seq(-3, 8, 5),  limits = c(-3, 8)) +
    scale_x_continuous(breaks = seq(-5, 25, 5), limits = c(-5, 30))

  ggsave(save_path, width = save_width, height = save_height,
         units = "cm", bg = "white", dpi = 1000)
  p
}

# Loop over hemispheres x runs
file_dir <- paste0(file.path(DATA_DIR, "Fig_S2"), "/")
for (hemi in c("Left", "Right")) {
  for (run in c("run-1", "run-2")) {
    file <- paste0(file_dir, hemi, "_", run, "_concat_vert.csv")
    save_path <- paste0(save_dir, hemi, "_", run, "_scatters.pdf")
    df <- read.table(file, header = TRUE, sep = ",")

    color_idx <- if (hemi == "Left") 1 else 2
    scatter_plot(
      df,
      col_x = "task_t", col_y = "predict_task_t",
      color_point = my_colors[color_idx],
      color_line  = my_colors[color_idx],
      fill_se     = my_colors[color_idx],
      stat_y_position = 7.8,
      fig_title = paste0(hemi, ": ", run),
      x_lab = "Actual SpPerAct",
      y_lab = "Predicted SpPerAct",
      save_path = save_path,
      save_width = 30, save_height = 30
    )
  }
}
