# process_tile.R 
# downloads .las or .laz file 
# saves .gpkg & .tif files in "subtile"_out ("subtile" being 25GN1 for example)
# sometimes necessary to run installpack.R once

library(sf)
library(dplyr)
library(httr)
library(lidR)
library(terra)
library(stringr)

# # Set a CRAN mirror explicitly
# options(repos = c(CRAN = "https://cloud.r-project.org"))

# # Install BiocManager (only once)
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# # Install EBImage from Bioconductor
# BiocManager::install("EBImage")


# Helper to filter noise
filter_noise = function(las, sensitivity) {
  p95 <- grid_metrics(las, ~quantile(Z, probs = 0.95), 10)
  las <- merge_spatial(las, p95, "p95")
  las <- filter_poi(las, Z < p95 * sensitivity)
  las$p95 <- NULL
  return(las)
}

# Updated function
download_and_process_tiles <- function(tile_list) {
  base_url <- "https://geotiles.citg.tudelft.nl/AHN4_T/tile_name.LAZ"
  save_directory <- "25GN1_out/tile_name.gpkg"
  chm_file_directory <- "25GN1_out/chm_tile_name.tif"

  classification_filter <- 1
  elevation_filter_min <- 2
  elevation_filter_max <- 40
  if (!dir.exists("25GN1_out")) dir.create("25GN1_out")

  for (tile_name in tile_list) {
    tryCatch({
      url <- str_replace(base_url, "tile_name", tile_name)
      save_path <- str_replace(save_directory, "tile_name", tile_name)
      chm_file_path <- str_replace(chm_file_directory, "tile_name", tile_name)

      if (file.exists(save_path)) {
        cat("Skipping", tile_name, "- already processed.\n")
        next
      }

      cat("Downloading", tile_name, "\n")

      # Download the LAZ file
      temp_laz <- tempfile(fileext = ".laz")
      temp_las <- tempfile(fileext = ".las")
      download_status <- tryCatch({
        download.file(url, destfile = temp_laz, mode = "wb", quiet = TRUE)
        TRUE
      }, error = function(e) {
        cat("Download failed for", tile_name, ":", conditionMessage(e), "\n")
        FALSE
      })

      if (!download_status || file.info(temp_laz)$size < 1e6) {
        cat("File incomplete or corrupt for", tile_name, "\n")
        next
      }


      # Convert to LAS using PDAL
      pdal_status <- system(paste("pdal translate", shQuote(temp_laz), shQuote(temp_las)))
      if (pdal_status != 0) {
        cat("PDAL conversion failed for", tile_name, "\n")
        next
      }

      # Read the LAS
      las <- readLAS(temp_las)
      if (is.empty(las)) {
        cat("LAS file is empty for", tile_name, "\n")
        next
      }
      # --- FIX: Scale 8-bit RGB values to 16-bit ---
      if (!is.null(las$Red) && max(las$Red, na.rm = TRUE) <= 255) {
        cat("Rescaling RGB values for", tile_name, "\n")
        las$Red   <- las$Red * 256
        las$Green <- las$Green * 256
        las$Blue  <- las$Blue * 256
      }

      # Preprocessing
      cat("Preprocessing", tile_name, "\n")
      cat("Normalizing...")
      las <- normalize_height(las, knnidw(k = 8, p = 2))
      cat("Done normalizing.")
      cat("Filtering...")
      las <- filter_poi(las, !(Classification == classification_filter & Z <= elevation_filter_min) &
                              !(Classification == classification_filter & Z >= elevation_filter_max))
      cat("Done filtering poi.")
      las <- filter_noise(las, sensitivity = 1.2)
      cat("Done filtering noise.")

      # Canopy Height Model (CHM)
      cat("Rasterizing canopy...")
      chm <- rasterize_canopy(las, res = 0.25, algorithm = p2r(0.2))
      cat("Done rasterizing canopy.")
      chm[is.na(chm)] <- 0
      w <- matrix(1, 3, 3)
      chm <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
      chm <- terra::focal(chm, w, fun = mean, na.rm = TRUE)
      cat("Done rasterizing. \n")

      # Tree segmentation
      cat("Watershedding... \n")
      algo <- lidR::watershed(chm, th = 3, tol = 2.5, ext = 1)
      las_watershed <- segment_trees(las, algo)
      trees <- filter_poi(las_watershed, !is.na(treeID))
      cat("Done watershed. \n")

      # Save CHM
      terra::writeRaster(chm, chm_file_path, overwrite = TRUE)
      cat("Saved CHM. \n")

      # Create tree crown polygons
      cat("Delineating trees... \n")
      hulls <- delineate_crowns(trees, type = "concave", concavity = 2, func = .stdmetrics)
      hulls <- st_as_sf(hulls)
      st_write(hulls, save_path, layer = "hulls", delete_dsn = TRUE)

      # Optionally save point cloud with treeIDs
      cat("Saving point cloud... \n")
      writeLAS(trees, paste0("25GN1_out/", tile_name, "_trees.laz"))

      # Clean up
      cat("Memory usage (Mb):", sum(gc()[,2]), "\n")
      # unlink(c(temp_laz, temp_las))
      # gc(); gc()

      # # Kill temporary raster tiles
      # terra::tmpFiles(current = TRUE, kill = TRUE)

      # # Remove temp LAS/LAZ
      # unlink(c(temp_laz, temp_las))
      # closeAllConnections()
      rm(hulls, trees, chm, las)
      gc(); gc()
      terra::tmpFiles(current = TRUE, remove = TRUE)
      unlink(c(temp_laz, temp_las))
      closeAllConnections()
      cat("Memory usage (Mb):", sum(gc()[,2]), "\n")
    }, error = function(e) {
      cat("Error processing", tile_name, ":", conditionMessage(e), "\n")
    })
  }
}

tile_list <- c("25GN1_01","25GN1_02","25GN1_03","25GN1_04","25GN1_05","25GN1_06","25GN1_07","25GN1_08","25GN1_09","25GN1_10")
download_and_process_tiles(tile_list)