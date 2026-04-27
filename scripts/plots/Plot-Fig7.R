# Fig 7: 4-group comparison panel for cognitive scores, brain averages, and laterality
# indices. Three rows (A: cognition, B: averages, C: LIs) combined with cowplot.
library(ggpubr)
library(tidyverse)
library(rstatix)
library(cowplot)

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

save_dir <- paste0(file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig7"), "/")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

my_colors <- c("#4472C4", "#E1683C", "#70AD47", "#9A5699")  # Nature-style palette

# --- Per-variable layout (Y limits, breaks, title) ---
plot_specs <- list(
  ReadEng_AgeAdj    = list(title = "ReadEng",          ylim = c(40, 185),  breaks = seq(75, 150, 25),
                           base_height_q = 0.75, line_spacing_q = 0.08),
  PicVocab_AgeAdj   = list(title = "PicVocab",         ylim = c(45, 190),  breaks = seq(50, 150, 50),
                           base_height_q = 0.65, line_spacing_q = 0.08),
  MeanSM_tPeak_mean = list(title = "SpPerActPeak_Avg", ylim = c(-2, 22),   breaks = seq(0, 15, 5),
                           base_height_q = 0.80, line_spacing_q = 0.12),
  NDI_mean          = list(title = "NDI_Avg",          ylim = c(0.33, 0.57), breaks = seq(0.36, 0.52, 0.04),
                           base_height_q = 0.80, line_spacing_q = 0.09),
  area_mean         = list(title = "Area_Avg",         ylim = c(0, 2200),  breaks = seq(500, 1500, 500),
                           base_height_q = 0.80, line_spacing_q = 0.10),
  MeanSM_tPeak_LI   = list(title = "SpPerActPeak_LI",  ylim = c(-0.7, 1.4), breaks = seq(-0.4, 0.8, 0.4),
                           base_height_q = 0.85, line_spacing_q = 0.10),
  area_LI           = list(title = "Area_LI",          ylim = c(-0.7, 1.4), breaks = seq(-0.4, 0.8, 0.4),
                           base_height_q = 0.85, line_spacing_q = 0.10)
)

# --- Build per-variable plots ---
plot_list_final <- list()
file_dir <- paste0(file.path(DATA_DIR, "Fig7"), "/")

for (col in names(plot_specs)) {
  spec <- plot_specs[[col]]
  data <- read.table(paste0(file_dir, col, "_4Groups.csv"),
                     header = TRUE, sep = ",")
  data$Groups <- as.factor(data$Groups)

  # --- Base plot: violin + box + jitter + mean trend ---
  point_plot <- ggplot(data, aes(x = Groups, y = value)) +
    geom_violin(aes(fill = Groups), alpha = 0.15, color = NA,
                scale = "width", width = 0.7, trim = FALSE) +
    geom_boxplot(aes(color = Groups), alpha = 0.3, fill = NA,
                 width = 0.2, outlier.shape = NA,
                 linewidth = 0.7, fatten = 2) +
    geom_jitter(aes(color = Groups), width = 0.12, height = 0,
                alpha = 0.15, size = 0.8, shape = 16) +
    stat_summary(fun = mean, geom = "point", shape = 18,
                 size = 2.8, color = "black", alpha = 0.95) +
    stat_summary(fun = mean, geom = "line", group = 1,
                 color = "#333333", linewidth = 0.8,
                 alpha = 0.5, linetype = "22") +
    scale_color_manual(values = my_colors) +
    scale_fill_manual(values = my_colors) +
    theme_classic(base_size = 10, base_family = "sans") +
    theme(
      plot.title         = element_text(size = 11, face = "bold", hjust = 0.5,
                                        margin = margin(b = 8, t = 5)),
      axis.line          = element_line(linewidth = 0.6, color = "#2C3E50"),
      axis.ticks         = element_line(linewidth = 0.5, color = "#2C3E50"),
      axis.ticks.length  = unit(0.15, "cm"),
      axis.text          = element_text(size = 9, color = "#2C3E50"),
      axis.text.x        = element_text(margin = margin(t = 4)),
      axis.text.y        = element_text(margin = margin(r = 4)),
      axis.title         = element_text(size = 10, color = "#2C3E50"),
      legend.position    = "none",
      plot.background    = element_rect(fill = "white", color = NA),
      panel.background   = element_rect(fill = "white", color = NA),
      panel.grid         = element_blank(),
      plot.margin        = margin(10, 12, 10, 12),
      plot.tag           = element_text(size = 13, face = "bold",
                                        hjust = 0, vjust = 1,
                                        margin = margin(b = 6, r = 4)),
      plot.tag.position  = c(0.01, 0.99)
    ) +
    ylab("") + xlab("")

  # --- Pairwise t-test with Bonferroni; place brackets only for significant pairs ---
  stat.test <- data %>%
    t_test(value ~ Groups, paired = FALSE, p.adjust.method = "bonferroni") %>%
    add_xy_position(x = "Groups", dodge = 0.8, step.increase = 0.12,
                    scales = "free", fun = "max")

  if (nrow(stat.test) > 0) {
    data_range  <- range(data$value, na.rm = TRUE)
    data_center <- mean(data_range)
    data_span   <- diff(data_range)
    base_height  <- data_center + data_span * spec$base_height_q
    line_spacing <- data_span * spec$line_spacing_q

    sig_idx <- which(stat.test$p.adj < 0.05)
    stat.test$y.position <- NA_real_
    if (length(sig_idx) > 0) {
      stat.test$y.position[sig_idx] <- seq(
        from = base_height, by = line_spacing, length.out = length(sig_idx)
      )
    }
  }

  plt <- point_plot +
    stat_pvalue_manual(
      stat.test, tip.length = 0, bracket.nudge.y = 0,
      hide.ns = TRUE, size = 2.8, label = "p.adj.signif",
      bracket.size = 0.35, color = "#2C3E50", step.increase = 0
    ) +
    scale_y_continuous(limits = spec$ylim, breaks = spec$breaks, expand = c(0, 0)) +
    ggtitle(spec$title)

  plot_list_final[[col]] <- plt
}

# --- Assemble panels A / B / C ---
plot_list_final[["ReadEng_AgeAdj"]]    <- plot_list_final[["ReadEng_AgeAdj"]]    + labs(tag = "A")
plot_list_final[["MeanSM_tPeak_mean"]] <- plot_list_final[["MeanSM_tPeak_mean"]] + labs(tag = "B")
plot_list_final[["MeanSM_tPeak_LI"]]   <- plot_list_final[["MeanSM_tPeak_LI"]]   + labs(tag = "C")

blank_plot <- ggplot() + theme_void()

row_A <- plot_grid(
  blank_plot, plot_list_final[["ReadEng_AgeAdj"]],
  plot_list_final[["PicVocab_AgeAdj"]], blank_plot,
  nrow = 1, rel_widths = c(0.25, 1, 1, 0.25),
  align = "h", axis = "tb"
)
row_B <- plot_grid(
  plot_list_final[["MeanSM_tPeak_mean"]],
  plot_list_final[["NDI_mean"]],
  plot_list_final[["area_mean"]],
  nrow = 1, rel_widths = c(1, 1, 1),
  align = "h", axis = "tb"
)
row_C <- plot_grid(
  blank_plot, plot_list_final[["MeanSM_tPeak_LI"]],
  plot_list_final[["area_LI"]], blank_plot,
  nrow = 1, rel_widths = c(0.25, 1, 1, 0.25),
  align = "h", axis = "tb"
)

final_plot <- plot_grid(row_A, row_B, row_C,
                        nrow = 3, rel_heights = c(1, 1, 1),
                        align = "v", axis = "lr")

# --- Save (PDF, journal-PDF, PNG, TIFF, EPS) ---
out_specs <- list(
  list(name = "Fig7_Final_Academic.pdf",         w = 20, h = 26, dpi = 600, bg = "white"),
  list(name = "Fig7_Final_Academic_Journal.pdf", w = 17, h = 22, dpi = 600, bg = "white"),
  list(name = "Fig7_Final_Academic.png",         w = 20, h = 26, dpi = 600, bg = "white"),
  list(name = "Fig7_Final_Academic.tiff",        w = 20, h = 26, dpi = 600, bg = "white",
       compression = "lzw")
)
for (s in out_specs) {
  args <- list(filename = paste0(save_dir, s$name), plot = final_plot,
               width = s$w, height = s$h, units = "cm",
               dpi = s$dpi, bg = s$bg)
  if (!is.null(s$compression)) args$compression <- s$compression
  do.call(ggsave, args)
}

# EPS not always available depending on cairo build
tryCatch({
  ggsave(paste0(save_dir, "Fig7_Final_Academic.eps"),
         plot = final_plot, width = 20, height = 26, units = "cm",
         dpi = 600, device = "eps")
}, error = function(e) {
  message("EPS export skipped: ", conditionMessage(e))
})
