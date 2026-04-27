# Fig 6: laterality-index analyses.
#   1. Paired raincloud comparing Left vs. Right hemisphere prediction score.
#   2. LI scatter plots (area / tPeak / thick) vs LI of prediction score.
#   3. Circular bar chart summarizing LI correlations across measures.
library(ggpubr)
library(tidyverse)
library(see)
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

save_dir <- paste0(file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig6"), "/")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# Custom paired-raincloud geom (https://yjunechoe.github.io/posts/2020-07-13-geom-paired-raincloud/)
source(file.path(SCRIPT_DIR, "geom_paired_raincloud.R"))

my_colors <- list("#404040", "#d62728")
LI_color  <- "#404040"

# --- 1. Prediction-score: Left vs Right (paired raincloud) ---
data <- read.table(file.path(DATA_DIR, "AllTaskCons-OwnCorr.csv"),
                   header = TRUE, sep = ",")

df <- data[, c("Subject", "LW_LPAC_MeanSM", "RW_RPAC_MeanSM")]
colnames(df)[colnames(df) == "LW_LPAC_MeanSM"] <- "Left SpPerAct"
colnames(df)[colnames(df) == "RW_RPAC_MeanSM"] <- "Right SpPerAct"

df_melt <- gather(df, "Contrasts", "Corr", "Left SpPerAct", "Right SpPerAct")
df_melt$Contrasts <- as.factor(df_melt$Contrasts)
df_ti <- as_tibble(df_melt)

bxp <- df_ti %>%
  arrange(Subject) %>%
  ggplot(aes(Contrasts, Corr, color = Contrasts)) +
  geom_paired_raincloud(aes(fill = Contrasts), alpha = 1) +
  geom_point(aes(group = Subject),
             position = position_nudge(c(0.15, -0.15)),
             alpha = 0.1, shape = 16, color = "black",
             show.legend = FALSE) +
  geom_line(aes(group = Subject), alpha = 0.1, color = "black",
            position = position_nudge(c(0.15, -0.15))) +
  geom_boxplot(aes(fill = Contrasts),
               position = position_nudge(c(0.07, -0.07)),
               alpha = 1, width = 0.04, outlier.shape = " ") +
  theme_pubr(base_size = 30) +
  theme(
    legend.position = "",
    legend.title    = element_blank(),
    plot.title      = element_text(size = 40, hjust = 0.5),
    axis.title.y    = element_text(size = 40, hjust = 0.5),
    axis.text.x     = element_text(size = 27, hjust = 0.5, vjust = 1)
  ) +
  xlab("") + ylab("Prediction Score") +
  scale_fill_manual("Contrasts",   values = c(my_colors[1], my_colors[2])) +
  scale_colour_manual("Contrasts", values = c("black", "black"))

# Overlay group-mean point + connecting line
df_mean <- df_ti %>%
  group_by(Contrasts) %>%
  summarize(average = mean(Corr)) %>%
  ungroup()

bxp +
  geom_point(data = df_mean,
             aes(x = Contrasts, y = average),
             alpha = 1, color = "black", size = 2, shape = 10,
             position = position_nudge(c(0.15, -0.15))) +
  geom_line(data = df_mean,
            aes(x = Contrasts, y = average, group = 1),
            alpha = 1, color = "black", size = 1,
            position = position_nudge(c(0.15, -0.15)))

ggsave(paste0(save_dir, "MeanSM_Corr_L_vs_R.pdf"),
       width = 20, height = 40, units = "cm", bg = "white", dpi = 1000)

# --- 2. LI scatter plots ---
li_x_labs <- c(
  area  = "Area_LI",
  tPeak = "SpPerActPeak_LI",
  thick = "Thick_LI",
  std   = "SpPerActSD_LI"
)

for (key_flag in c("area", "tPeak", "thick")) {
  file <- file.path(DATA_DIR, "Fig6",
                    paste0(key_flag, "_LI_vs_MeanSM_Corr_LI_GLM.csv"))
  df <- read.table(file, header = TRUE, sep = ",")
  x_lab <- li_x_labs[[key_flag]]

  p <- ggplot(df, aes(x = x, y = adjusted_y)) +
    geom_point(size = 1, alpha = 0.5, color = LI_color) +
    geom_smooth(method = lm, color = LI_color, fill = LI_color, se = TRUE) +
    stat_cor(label.y = 1.6, r.accuracy = 0.01, size = 10) +
    theme_pubr(base_size = 30) +
    theme(
      plot.title   = element_text(size = 25, hjust = 0.5),
      axis.title.y = element_text(size = 35, hjust = 0.5),
      axis.title.x = element_text(size = 35, hjust = 0.5, vjust = 1)
    ) +
    xlab(x_lab) + ylab("LI of the PS") + ggtitle("")

  ggsave(paste0(save_dir, "LI_", key_flag, "_PredCorr.pdf"),
         plot = p, width = 20, height = 20, units = "cm",
         bg = "white", dpi = 1000)
}

# --- 3. Circular bar of LI correlations ---
data <- read.table(file.path(DATA_DIR, "Fig6", "LI_Measures_vs_PredCorr-Corr.csv"),
                   header = TRUE, sep = ",")
li_idx_map <- c(
  MeanSM_tPeak_LI = "SpPerActPeak_LI",
  MeanSM_std_LI   = "SpPerActSD_LI",
  area_LI         = "Area_LI",
  ISO_LI          = "ISO_LI",
  ODI_LI          = "ODI_LI",
  NDI_LI          = "NDI_LI",
  WIN_LI          = "WIN_LI",
  myelin_LI       = "Myelin_LI",
  thick_LI        = "Thick_LI"
)
for (k in names(li_idx_map)) data$Indices[data$Indices == k] <- li_idx_map[[k]]
data$Indices <- as.factor(data$Indices)
df <- data

LI_color <- "#4A3C6B"
plt <- ggplot(df) +
  geom_hline(aes(yintercept = y),
             data.frame(y = c(-0.3, -0.2, 0, 0.2, 0.4)),
             color = "lightgrey") +
  geom_col(aes(x = reorder(str_wrap(Indices, 5), Corr), y = Corr),
           fill = LI_color, color = "black",
           position = position_dodge(0.5), width = 0.5, alpha = 0.9) +
  geom_segment(aes(x    = reorder(str_wrap(Indices, 5), Corr),
                   y    = -0.3,
                   xend = reorder(str_wrap(Indices, 5), Corr),
                   yend = 0.4,
                   alpha = 0.5),
               linetype = "dashed", color = "gray12") +
  coord_polar()

x_loc <- 8.7; font_size <- 7.5
plt <- plt +
  annotate("text", x = x_loc, y = -0.16, label = "-0.2",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.04, label = "0",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.24, label = "0.2",
           color = "gray12", size = font_size) +
  annotate("text", x = x_loc, y =  0.44, label = "0.4",
           color = "gray12", size = font_size) +
  theme(
    axis.title         = element_blank(),
    axis.ticks         = element_blank(),
    axis.text.y        = element_blank(),
    axis.text.x        = element_blank(),
    legend.position    = "",
    panel.background   = element_rect(fill = "white", color = "white"),
    panel.grid         = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.title       = element_blank()
  ) +
  geom_textpath(aes(x = str_wrap(Indices, 5), y = 0.45,
                    label = str_wrap(Indices, 5)),
                hjust = 0.5, vjust = 2, colour = "gray12", size = 9)

ggsave(paste0(save_dir, "LI_BrainMeasures_PredCorr_circular.pdf"),
       plot = plt, width = 20, height = 20, units = "cm",
       bg = "white", dpi = 1000)
