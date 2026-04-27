# Fig 5 (circular bar): brain measures vs. prediction score, polished version.
library(tidyverse)
library(ggplot2)
library(stringr)
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

my_colors <- list("#404040", "#d62728")

# --- Load and rename ---
data <- read.table(file.path(DATA_DIR, "Fig5", "Measures_vs_PredCorr-Corr.csv"),
                   header = TRUE, sep = ",")
data$PAC[data$PAC == "LPAC"] <- "Left peri-Sylvian AC"
data$PAC[data$PAC == "RPAC"] <- "Right peri-Sylvian AC"

idx_map <- c(
  MeanSM_std   = "SpPerActSD",
  MeanSM_tPeak = "SpPerActPeak",
  myelin       = "Myelin",
  thick        = "Thick",
  area         = "Area",
  g            = "g-factor"
)
for (k in names(idx_map)) data$Indices[data$Indices == k] <- idx_map[[k]]

data$PAC      <- as.factor(data$PAC)
data$Features <- as.factor(data$Features)
data$Indices  <- as.factor(data$Indices)
df <- data

# --- Circular bar ---
plt <- ggplot(df) +
  geom_hline(aes(yintercept = y),
             data.frame(y = c(-0.3, -0.2, 0, 0.2, 0.4, 0.6)),
             color = "lightgrey") +
  geom_col(aes(x = reorder(str_wrap(Indices, 5), Corr),
               y = Corr, fill = PAC, color = PAC),
           position = "dodge2", show.legend = TRUE, alpha = 0.9) +
  geom_segment(aes(x    = reorder(str_wrap(Indices, 5), Corr),
                   y    = -0.3,
                   xend = reorder(str_wrap(Indices, 5), Corr),
                   yend = 0.6),
               linetype = "dashed", color = "gray12", alpha = 0.5) +
  coord_polar()

# --- Y-axis radial labels ---
x_loc <- 12.7
font_size <- 7
plt <- plt +
  annotate("text", x = 11.7, y = -0.16, label = "-0.2",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.04, label = "0",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.24, label = "0.2",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.44, label = "0.4",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.64, label = "0.6",
           color = "gray12", size = font_size) +
  theme(
    axis.title           = element_blank(),
    axis.ticks           = element_blank(),
    axis.text.y          = element_blank(),
    axis.text.x          = element_blank(),
    legend.position      = c(0.85, 0.95),
    panel.background     = element_rect(fill = "white", color = "white"),
    panel.grid           = element_blank(),
    panel.grid.major.x   = element_blank(),
    legend.title         = element_blank(),
    legend.text          = element_text(size = 15),
    plot.margin          = margin(0, 0, 0, 0, "pt")
  ) +
  geom_textpath(aes(x = str_wrap(Indices, 5), y = 0.65,
                    label = str_wrap(Indices, 5)),
                hjust = 0.5, vjust = 2, colour = "gray12", size = 7) +
  scale_fill_manual("PAC",   values = c(my_colors[1], my_colors[2])) +
  scale_colour_manual("PAC", values = c("black", "black"))

# --- Save ---
ggsave(paste0(save_dir, "BrainMeasures_PredCorr_circular.pdf"),
       width = 20, height = 20, units = "cm", bg = "white", dpi = 1000)
