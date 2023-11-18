# new shap data processing....
# processing one single shapley matrix and try to identify any spots above 0.95 criteria.
args <- commandArgs(trailingOnly = TRUE)
if(length(args)!=2){
  print("Usage: Rscript *.R <full path to input shapley matrix folder> <sample name>")
  quit()
}else{
  # default threshold 0.95
  library(reshape2)
  threshold=0.95
  path=args[1]
  sample=args[2]
  file=list.files(path=path,pattern = paste0(sample,".*.txt"))	
  #args="/research/bsi/projects/breast/s301449.LARdl/processing/MOLI_circos/TCGA/BRCA/results/naresh_results/cnv_only/shapely_heatmaps/TCGA.AC.A3TM.01A.png.shap_sample1000.txt"
  mat <- read.table(paste(path,file,sep="/"),header=T,sep="\t",row.names = 1,stringsAsFactors = F)
  uvals <- unique(unlist(mat))
  # find the cutoffs on the upper and lower 95th quantile of unique values
  cutoff=quantile(uvals,probs = c(1-threshold,threshold))
  
  # obtain only border, for col block detection
  detect_runs_border <- function(x) {
    rle_x <- rle(x)
    values <- rle_x$values
    lengths <- rle_x$lengths
    runs <- which(lengths >= 1) # Adjust threshold as needed
    borders <- unname(append(c(0, cumsum(lengths))[runs],256)+1)
    return(borders)
  }
  
  # obtain both border and value, for row block detection
  detect_runs_bv <- function(x) {
    rle_x <- rle(x)
    values <- rle_x$values
    lengths <- rle_x$lengths
    runs <- which(lengths >= 1) # Adjust threshold as needed
    borders <- unname(append(c(0, cumsum(lengths))[runs],256)+1)
    bvs <- paste(borders,c(values,"null"),sep=":")
    return(bvs)
  }
  
  # Detect blocks in rows 
  rows <- unname(apply(mat, 1,detect_runs_bv ))
  
  # Detect blocks in columns
  cols <- unname(detect_runs_border(sapply(rows,function(x) paste(x,collapse = ","))))
  
  # Loop over blocks and find border coordinates
  blocks <- data.frame()
  for(j in 1:(length(cols)-1) ){
    
    # from line cols[j] to line cols[j+1]-1, which are the same rows.
    rows_j = rows[[ cols[j] ]]
    rows_j_b=as.numeric(sapply(rows_j,function(x) unlist(strsplit(x,":"))[1]))
    rows_j_v=sapply(rows_j,function(x) unlist(strsplit(x,":"))[2])
    for(k in 1:(length(rows_j)-1)){
      x_min <- rows_j_b[k] 
      x_max <- rows_j_b[k+1]-1  
      y_min <- cols[j]
      y_max <- cols[j+1]-1 
      #print(ifelse(mat[y_min,x_min]-as.numeric(rows_j_v[k]) <10^15 ,"yes","no"))
      if (x_max - x_min >= 2 && y_max - y_min >= 2) { # Adjust minimum block size as needed
        blocks <- rbind(blocks, c(y_min, y_max,x_min,x_max, mat[y_min,x_min]))
      }
    }
  }
  
  # x corresponding to col, y corresponding to row
  colnames(blocks) <- c("row_min","row_max","col_min","col_max","value")
  
  
  blocks$label <- ifelse(blocks$value>cutoff[2],"pos",ifelse(blocks$value<cutoff[1],"neg","null"))
  blocks.selected <- blocks[blocks$label!="null",]
  
  write.table(blocks.selected,file=paste0(sample,".peaks.txt"),quote=F,sep="\t",row.names = F)
  
  
  
}

