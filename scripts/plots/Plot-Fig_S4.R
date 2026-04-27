# Fig S4: ridge plots of prediction-score distributions across all task contrasts
# (language-based correlation; 3 variants: simple / optimized / grouped-by-domain).
library(tidyverse)
library(ggridges)
library(hrbrthemes)
library(viridis)
library(showtext)

tryCatch({
  font_add("Arial", "/Library/Fonts/Arial.ttf")
  showtext_auto()
}, error = function(e) {
  cat("Font loading failed; using default font.\n")
})

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

save_dir <- file.path(PROJECT_ROOT, "papers", "figures", "raw", "Fig_S4")
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

# --- Load and rename columns ---
data_file <- file.path(DATA_DIR, "AllTaskCons-LanguageBasedCorr.csv")
if (!file.exists(data_file)) stop("Input data not found: ", data_file)
data <- read.csv(data_file, header = TRUE)

# Strip LW_/RW_ prefixes
colnames(data) <- gsub("^LW_LPAC", "LPAC", colnames(data))
colnames(data) <- gsub("^RW_RPAC", "RPAC", colnames(data))

# Pretty rename: <hemisphere> <DOMAIN-CONTRAST>
rename_map <- c(
  LPAC_MeanSM = "Left LAN-SpPerAct",   RPAC_MeanSM = "Right LAN-SpPerAct",
  LPAC_Story_Math = "Left LAN-SpComAct", RPAC_Story_Math = "Right LAN-SpComAct",
  LPAC_RAD = "Left SOC-RAD",           RPAC_RAD = "Right SOC-RAD",
  LPAC_TOM = "Left SOC-TOM",           RPAC_TOM = "Right SOC-TOM",
  LPAC_RAD_TOM = "Left SOC-RAD_TOM",   RPAC_RAD_TOM = "Right SOC-RAD_TOM",
  LPAC_FACE = "Left EMO-FACE",         RPAC_FACE = "Right EMO-FACE",
  LPAC_SHAPE = "Left EMO-SHAPE",       RPAC_SHAPE = "Right EMO-SHAPE",
  LPAC_F_S = "Left EMO-FACE_SHAPE",    RPAC_F_S = "Right EMO-FACE_SHAPE",
  LPAC_PUNISH = "Left GAM-PUNISH",     RPAC_PUNISH = "Right GAM-PUNISH",
  LPAC_REWARD = "Left GAM-REWARD",     RPAC_REWARD = "Right GAM-REWARD",
  LPAC_PUNISH_REWARD = "Left GAM-PUNISH_REWARD",
  RPAC_PUNISH_REWARD = "Right GAM-PUNISH_REWARD"
)
for (old in names(rename_map)) {
  if (old %in% colnames(data)) colnames(data)[colnames(data) == old] <- rename_map[[old]]
}
if ("Subject" %in% colnames(data)) data$Subject <- NULL

# --- Long format, sorted by median ---
data_long <- data %>%
  pivot_longer(cols = everything(), names_to = "text", values_to = "value") %>%
  filter(!is.na(value))

task_order <- data_long %>%
  group_by(text) %>%
  summarise(median_val = median(value, na.rm = TRUE)) %>%
  arrange(median_val) %>%
  pull(text)

data_long <- data_long %>% mutate(text = factor(text, levels = task_order))

# --- Variant 1: simple ridge plot (viridis) ---
p1 <- ggplot(data_long, aes(x = value, y = text, fill = text)) +
  geom_density_ridges(alpha = 0.6, rel_min_height = 0.001,
                      bandwidth = 0.08, from = -1, to = 1) +
  scale_fill_viridis(discrete = TRUE) +
  scale_color_viridis(discrete = TRUE) +
  scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
                     expand = c(0, 0)) +
  theme_ipsum(grid = FALSE, base_size = 20) +
  theme(
    legend.position = "none",
    panel.spacing   = unit(0.1, "lines"),
    panel.grid      = element_blank(),
    axis.ticks.x    = element_blank(),
    plot.title      = element_text(size = 25, hjust = 0.5),
    axis.title.y    = element_text(size = 18, hjust = 0.5),
    axis.title.x    = element_text(size = 25, hjust = 0.5),
    axis.text.y     = element_text(size = 12, hjust = 1)
  ) +
  labs(title = "", x = "PS", y = "")

ggsave(file.path(save_dir, "AllCons_Corr_own.pdf"),
       plot = p1, width = 35, height = 20, units = "cm",
       dpi = 1000, bg = "white")

# --- Variant 2: optimized (gradient palette + median lines) ---
n_tasks <- length(unique(data_long$text))
ridge_colors <- colorRampPalette(
  c("#2E3440", "#5E81AC", "#88C0D0", "#EBCB8B", "#BF616A")
)(n_tasks)

p2 <- ggplot(data_long, aes(x = value, y = text, fill = text)) +
  geom_density_ridges(alpha = 0.75, scale = 1.2, rel_min_height = 0.01,
                      bandwidth = 0.08, from = -1, to = 1,
                      quantile_lines = TRUE, quantiles = 2) +
  scale_fill_manual(values = ridge_colors) +
  scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
                     expand = c(0, 0)) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position       = "none",
    plot.title            = element_text(size = 18, hjust = 0.5, face = "bold"),
    axis.title.x          = element_text(size = 16, face = "bold"),
    axis.title.y          = element_blank(),
    axis.text.y           = element_text(size = 10),
    axis.text.x           = element_text(size = 12),
    panel.grid.major.x    = element_line(color = "gray92", size = 0.3),
    panel.grid.major.y    = element_blank(),
    panel.grid.minor      = element_blank(),
    plot.background       = element_rect(fill = "white", color = NA),
    plot.margin           = margin(1, 1, 1, 1, "cm")
  ) +
  labs(x = "Prediction Score", y = NULL,
       title = "Task Contrasts Performance Distribution")

ggsave(file.path(save_dir, "AllCons_Ridge_optimized.pdf"),
       plot = p2, width = 14, height = 10, dpi = 300)

# --- Variant 3: grouped by task domain ---
data_categorized <- data_long %>%
  mutate(
    Category = case_when(
      grepl("LAN", text) ~ "Language",
      grepl("SOC", text) ~ "Social",
      grepl("EMO", text) ~ "Emotion",
      grepl("GAM", text) ~ "Gambling",
      TRUE ~ "Other"
    ),
    Hemisphere = ifelse(grepl("^Left", text), "Left", "Right")
  )

task_order_cat <- data_categorized %>%
  group_by(text, Category) %>%
  summarise(median_val = median(value, na.rm = TRUE), .groups = "drop") %>%
  arrange(Category, median_val) %>%
  pull(text)
data_categorized$text <- factor(data_categorized$text, levels = task_order_cat)

category_colors <- list(
  Language = c("#E63946", "#F77F00"),
  Social   = c("#06AED5", "#086788"),
  Emotion  = c("#7209B7", "#B5179E"),
  Gambling = c("#2A9D8F", "#264653")
)
color_mapping <- vapply(task_order_cat, function(task) {
  cat_name <- as.character(unique(data_categorized$Category[data_categorized$text == task]))
  pair <- if (cat_name %in% names(category_colors)) category_colors[[cat_name]] else c("#999999", "#666666")
  if (grepl("^Left", task)) pair[1] else pair[2]
}, character(1))

p3 <- ggplot(data_categorized, aes(x = value, y = text, fill = text)) +
  geom_density_ridges(alpha = 0.8, scale = 1.1,
                      bandwidth = 0.08, from = -1, to = 1) +
  scale_fill_manual(values = color_mapping) +
  scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.5),
                     expand = c(0, 0)) +
  facet_grid(Category ~ ., scales = "free_y", space = "free_y") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position    = "none",
    strip.text.y       = element_text(angle = 0, size = 20),
    strip.background   = element_rect(fill = "gray95", color = "gray80"),
    plot.title         = element_text(size = 25, hjust = 0.5, face = "bold"),
    axis.title.x       = element_text(size = 25),
    axis.text.y        = element_text(size = 15),
    axis.text.x        = element_text(size = 20),
    panel.spacing.y    = unit(0.5, "lines")
  ) +
  labs(x = "Prediction Score", y = NULL, title = NULL)

ggsave(file.path(save_dir, "AllCons_Ridge_grouped.pdf"),
       plot = p3, width = 12, height = 14, dpi = 300)

cat("\nDone. Outputs in:", save_dir, "\n")
