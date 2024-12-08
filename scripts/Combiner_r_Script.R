library(tidyverse)

header.true <- function(df) {
  names(df) <- as.character(unlist(df[1,]))
  df[-1,]
}

# REPLACE PATH WITH OWN PATH


# List all TSV files in the directory
setwd("C:\\Users\\dwarf\\Dataproject\\data_normalized\\2mers_extend200")
tsv_files <- list.files(pattern = "\\.tsv$")

# Sort the list of TSV files
#tsv_files <- sort(tsv_files)

# Initialize an empty list to store data frames
data_list <- list()

# Loop through each TSV file, read it, and store it in the list
for (file in tsv_files) {
  data <- read_tsv(file)
  data <- as.data.frame(t(data))
  data <- header.true(data)
  data <- data[,-1]
  data_list[[file]] <- data
}

# Combine all data frames into a single data frame
combined_2mers <- do.call(rbind, data_list)

# List all TSV files in the directory
setwd("C:\\Users\\dwarf\\Dataproject\\data_normalized\\4mers_extend200")
tsv_files <- list.files(pattern = "\\.tsv$")

# Sort the list of TSV files
#tsv_files <- sort(tsv_files)

# Initialize an empty list to store data frames
data_list <- list()

# Loop through each TSV file, read it, and store it in the list
for (file in tsv_files) {
  data <- read_tsv(file)
  data <- as.data.frame(t(data))
  data <- header.true(data)
  data <- data[,-1]
  data_list[[file]] <- data
}

# Combine all data frames into a single data frame
combined_4mers <- do.call(rbind, data_list)

# List all TSV files in the directory
setwd("C:\\Users\\dwarf\\Dataproject\\data_normalized\\6mers_extend200")
tsv_files <- list.files(pattern = "\\.tsv$")

# Sort the list of TSV files
#tsv_files <- sort(tsv_files)

# Initialize an empty list to store data frames
data_list <- list()

# Loop through each TSV file, read it, and store it in the list
for (file in tsv_files) {
  data <- read_tsv(file)
  data <- as.data.frame(t(data))
  data <- header.true(data)
  data <- data[,-1]
  data_list[[file]] <- data
}

# Combine all data frames into a single data frame
combined_6mers <- do.call(rbind, data_list)

write_tsv(combined_2mers,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_2mers.tsv")
write_tsv(combined_4mers,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_4mers.tsv")
write_tsv(combined_6mers,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_6mers.tsv")
meta_data <- read_tsv("C:\\Users\\dwarf\\Dataproject\\ctDNA_frag_meth\\metadata.tsv", col_names = FALSE)

combined_2mers_meth <- combined_2mers %>% slice(which(row_number()%% 2 == 1))
combined_2mers_unmeth <- tail(combined_2mers,-1)
combined_2mers_unmeth <- combined_2mers_unmeth %>% slice(which(row_number()%% 2 == 1))

combined_4mers_meth <- combined_4mers %>% slice(which(row_number()%% 2 == 1))
combined_4mers_unmeth <- tail(combined_4mers,-1)
combined_4mers_unmeth <- combined_4mers_unmeth %>% slice(which(row_number()%% 2 == 1))

combined_6mers_meth <- combined_6mers %>% slice(which(row_number()%% 2 == 1))
combined_6mers_unmeth <- tail(combined_6mers,-1)
combined_6mers_unmeth <- combined_6mers_unmeth %>% slice(which(row_number()%% 2 == 1))

meta_data_fin <- rbind(c("background","background"),meta_data)
meta_data_sort <- meta_data_fin[order(meta_data_fin$X1),]

combined_2mers_meth<- combined_2mers_meth %>% mutate("cancer" = meta_data_sort$X2)
combined_2mers_unmeth <- combined_2mers_unmeth %>% mutate("cancer" = meta_data_sort$X2)
combined_4mers_meth <- combined_4mers_meth %>% mutate("cancer" = meta_data_sort$X2)
combined_4mers_unmeth <- combined_4mers_unmeth %>% mutate("cancer" = meta_data_sort$X2)
combined_6mers_meth <- combined_6mers_meth %>% mutate("cancer" = meta_data_sort$X2)
combined_6mers_unmeth <- combined_6mers_unmeth %>% mutate("cancer" = meta_data_sort$X2)

write_tsv(combined_2mers_meth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_2mers_meth.tsv")
write_tsv(combined_4mers_meth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_4mers_meth.tsv")
write_tsv(combined_6mers_meth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_6mers_meth.tsv")
write_tsv(combined_6mers_unmeth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_6mers_unmeth.tsv")
write_tsv(combined_4mers_unmeth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_4mers_unmeth.tsv")
write_tsv(combined_2mers_unmeth,"C:\\Users\\dwarf\\Dataproject\\data_combined\\combined_2mers_unmeth.tsv")

