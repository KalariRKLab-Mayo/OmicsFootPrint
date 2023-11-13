# using TCGA lung as demo. 
# Usage: Rscript circos.plot.***.R id ( id is a number from 1:ncol(exprs))

#setwd("/infodev1/infoderm/Projects/Naresh/scripts/misc/OmicsFootPrint/sample_data")
suppressMessages(library(circlize))
library(parallel)

args <-  commandArgs(trailingOnly = TRUE)
#id <- as.numeric(args[1])
dir <- as.character(args[1])
setwd(dir)

#load anno.txt  cnv.txt  exprs.txt  file_list.txt  phospho.txt  totalp.txt
totalp<-read.table("totalp.txt",sep="\t",header=T,stringsAsFactors=F)
exprs<-read.table("exprs.txt",sep="\t",header=T,stringsAsFactors=F)
cnv<-read.table("cnv.txt",sep="\t",header=T,stringsAsFactors=F)
phospho<-read.table("phospho.txt",sep="\t",header=T,stringsAsFactors=F)
anno<-read.table("anno.txt",sep="\t",header=T,stringsAsFactors=F)
# set output path
path=dir




process_sample <- function(id) {
# nm is the name of the sample  
nm=colnames(exprs)[id]
print(nm)

print(paste0("we are priting: #",id,". The id is ",nm))

# if nm not a 01 tumor, skip. if nm doesn't have subtype info, skip 
#if(substr(nm,14,15)!="01"  ){
#	print(paste0(nm, " is not tumor!"))
#	quit()
#}
if(! nm %in% colnames(phospho)){
	print(paste0(nm, " is not in RPPA!"))
	quit()
}

cnv$sample <- gsub("-",".",cnv$sample)
# cnv format
# sample  Chrom  Start End        value
# TCGA-BH-A1FN-01A     1  62920   16689764       0.0681
# TCGA-BH-A1FN-01A     1  16718963        16721984        -1.2849

# subset cnv for current sample
cnv.nm <- cnv[cnv$sample==nm,]


png(paste0(path,"/",nm,".png"), units = "px",width=1024,height = 1024) 

circos.initializeWithIdeogram(species='hg38', chromosome.index = paste0("chr", c(1:22, "X", "Y")),plotType = NULL)
print("plot cnv...")

# layer 1: CNV
circos.trackPlotRegion(ylim = c(-9.5, 6.5), bg.border = "#FFFFFF", bg.col ="#FFFFFF",
                       track.height = 0.2, panel.fun = function(x, y) {
                         xlim = get.cell.meta.data("xlim")
                         ylim = get.cell.meta.data("ylim")
                         
                         xrange = get.cell.meta.data("xrange")
                       })

for(i in 1:nrow(cnv.nm)){
	# print(is.numeric(cnv.nm[i,"End"]))
	# print(paste("chr",cnv.nm[i,"Chrom"]))
	if(cnv.nm[i,"Chrom"]!="Y"){
        	circos.rect(xleft=cnv.nm[i,"Start"], ybottom=0,xright=cnv.nm[i,"End"],ytop=cnv.nm[i,"value"], sector.index=paste0("chr",cnv.nm[i,"Chrom"]),col="black")
	}
}

print("plot RPPA...")

#layer 2: RPPA
# notice that there would be NA in lung data. 
lower=floor(min(phospho,na.rm=T))
upper=ceiling(max(totalp,na.rm=T))
circos.track(ylim = c(lower,upper), bg.border = "#FFFFFF", bg.col ="#FFFFFF",
             track.height = 0.2, panel.fun = function(x, y) {
               xlim = get.cell.meta.data("xlim")
               ylim = get.cell.meta.data("ylim")
               
               xrange = get.cell.meta.data("xrange")
             })

id=match(nm,colnames(phospho))
#for(i in 1:2){
for(i in 1:nrow(phospho)){ 
  if(!is.na(phospho[i,id])){
  	circos.barplot(value =phospho[i,id], pos=0.5*(anno[match(rownames(phospho)[i],anno$gene),"chromStart"]+ anno[match(rownames(phospho)[i],anno$gene),"chromEnd"]),sector.index = anno[match(rownames(phospho)[i],anno$gene),"chrom"],col="red",bar_width = 0.6)
  }
}

for(i in 1:nrow(totalp)){ 
  if(!is.na(totalp[i,id])){
  	circos.barplot(value =totalp[i,id], pos=0.5*(anno[match(rownames(totalp)[i],anno$gene),"chromStart"]+ anno[match(rownames(totalp)[i],anno$gene),"chromEnd"]),sector.index = anno[match(rownames(totalp)[i],anno$gene),"chrom"],col="green",bar_width = 0.6)
  }	
}

print("plot exprs...")
# layer 3: expression 
circos.track(ylim = c(0, ceiling(max(exprs) )), bg.border = "#FFFFFF", bg.col ="#FFFFFF",
			 track.height = 0.2, panel.fun = function(x, y) {
			   xlim = get.cell.meta.data("xlim")
			   ylim = get.cell.meta.data("ylim")
			   
			   xrange = get.cell.meta.data("xrange")
			 })


for(i in 1:nrow(exprs)){ 
  if(rownames(exprs)[i] %in% rownames(anno) & ! anno[rownames(exprs)[i],"chrom"] %in% c("chrY","chrM") ){
	
	#circos.points( x= entrezID.chrloc.df[rownames(exprs)[i],"pos"], y=exprs[i,j], sector.index = paste0("chr",entrezID.chrloc.df[rownames(exprs)[i],"chr"]),cex=1,pch=".")
	circos.barplot(value =exprs[i,nm], pos=0.5*(anno[rownames(exprs)[i],"chromStart"]+ anno[rownames(exprs)[i],"chromEnd"]),sector.index = anno[rownames(exprs)[i],"chrom"],col="black",bar_width = 0.6)
  }
}


dev.off()
    

  
  
  #    circos.points( x= entrezID.chrloc.df[rownames(exprs)[i],"pos"], y=exprs[i,1], sector.index = paste0("chr",entrezID.chrloc.df[rownames(exprs)[i],"chr"]) ,col="green",cex=1,pch=".")
  
  
}

# Define the range of ids to process (replace with actual number of columns in exprs)
ids_to_process <- 1:ncol(exprs)

# Use detectCores() to find out the number of cores on your machine
no_cores <- detectCores() - 1 # reserve one core for system processes

# Run the process_sample function in parallel over the ids
# This will return a list of results
results <- mclapply(ids_to_process, process_sample, mc.cores = no_cores)
