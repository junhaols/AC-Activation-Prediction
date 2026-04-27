# Fig 2D: paired violin plots (Left vs. Right peri-Sylvian AC) for the validation dataset.
library(tidyverse)
library(ggpubr)

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

# HCP-aligned palette (red = left, blue = right)
hcp_palette <- c(
  "Left peri-\nSylvian AC"  = "#E64B35",
  "Right peri-\nSylvian AC" = "#4DBBD5"
)

for (run in c("run-1", "run-2")) {
  file <- file.path(DATA_DIR, "Fig2", paste0("All_subs_", run, "_corr.csv"))
  data1 <- read.table(file, header = TRUE, sep = ",")

  data1 <- data1 %>%
    mutate(
      PAC = case_when(
        PAC == "LPAC" ~ "Left peri-\nSylvian AC",
        PAC == "RPAC" ~ "Right peri-\nSylvian AC"
      ),
      PAC = factor(PAC, levels = c("Left peri-\nSylvian AC", "Right peri-\nSylvian AC")),
      Subject = as.factor(Subject)
    )

  p_violin_paired <- ggplot(data1, aes(x = PAC, y = Corr)) +
    geom_line(aes(group = Subject), color = "grey70", alpha = 0.8) +
    geom_violin(aes(fill = PAC), trim = FALSE, alpha = 0.5,
                draw_quantiles = c(0.5)) +
    geom_point(aes(color = PAC), size = 2, alpha = 0.9) +
    scale_fill_manual(values = hcp_palette) +
    scale_color_manual(values = hcp_palette) +
    labs(title = paste("Validation dataset", run),
         x = "", y = "Prediction Score") +
    ylim(-0.6, 1.4) +
    theme_classic(base_size = 45) +
    theme(
      plot.title      = element_text(hjust = 0.5, size = rel(1.5)),
      axis.title      = element_text(size = rel(1.2)),
      axis.text       = element_text(color = "black", size = rel(1)),
      axis.line       = element_line(linewidth = 1.5),
      axis.ticks      = element_line(linewidth = 1.5),
      legend.position = "none"
    )

  save_file <- file.path(save_dir, paste0("Validation-", run, "_violin_paired.pdf"))
  ggsave(save_file, plot = p_violin_paired,
         width = 14, height = 16, units = "cm",
         bg = "white", dpi = 1000)
}
