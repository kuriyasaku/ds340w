# Modified Implementation Platform for Large Pathology Image Patch Extraction

## Overview

This project focuses on improving the preprocessing pipeline for very large pathology images.  
In the original implementation, patch extraction from whole-slide pathology images was slower and less practical for large-scale experiments.  
To improve this part of the pipeline, I replaced the previous image processing approach with **pyvips**.

The main goal of this modification is not to introduce a new learning model, but to improve the **implementation platform** used for patch extraction.  
This change makes the preprocessing step faster and more suitable for extremely large pathology images.

---

## Motivation

Whole-slide pathology images are very large and can be difficult to process efficiently.  
In the earlier version of the pipeline, reading and cutting these images into patches was time-consuming and not ideal for scaling to many slides.

After testing different tools, I found that **pyvips** performs much better for this task.  
It is faster and more practical for handling large image files.

---

## Novelty / Contribution

The novelty of this work is an **implementation-level improvement** in the pathology image preprocessing pipeline.

### What was changed?
- Replaced the previous patch extraction backend with **pyvips**
- Used pyvips for reading and cutting large pathology images into patches

### Why is this useful?
- Faster patch extraction
- Better suited for very large pathology images
- More practical for large-scale preprocessing
- Improves the efficiency of the overall pipeline

This project does not mainly claim a new model-level contribution.  
Instead, it presents a **modified implementation platform** that improves the preprocessing stage.

---

## Method

### Previous approach
The earlier pipeline used a different image processing method for reading and extracting patches from pathology images.

### New approach
The updated pipeline uses **pyvips** as the main backend for patch extraction.

### Expected advantage
Because pyvips is designed for efficient large-image processing, it can reduce preprocessing time and make the workflow more scalable.

---

## Comparison

A simple comparison can be used to show the benefit of this modification:

| Method | Patch Extraction Speed | Scalability | Practicality for Large Images |
|--------|------------------------|-------------|-------------------------------|
| Previous method | Slower | Limited | Less practical |
| pyvips-based method | Faster | Better | More practical |

> You can replace this table with real experimental numbers if available.

For example, you may report:
- time used per slide
- total time for a batch of slides
- memory usage or stability observations

---

## Project Scope

This work focuses on the **preprocessing / patch extraction module** of the pipeline.  
The purpose is to improve the technical implementation of one modular block from the previous assignment.

This is aligned with the course goal of modifying at least one part of the implementation platform and introducing novelty or creativity in the workflow.

---

## Tools Used

- Python
- pyvips
- Other supporting Python libraries for file handling and patch organization

---

## Conclusion

By replacing the original patch extraction method with **pyvips**, this project improves the implementation platform for large pathology image preprocessing.  
This modification makes the workflow faster and more practical, especially when working with very large whole-slide images.

The main contribution of this work is a more efficient preprocessing pipeline, which can support later experiments more effectively.

---

## Future Work

Possible future improvements include:
- adding full benchmark results on multiple slides
- comparing memory usage across methods
- integrating the faster extraction pipeline into the full training workflow
- testing the method on larger pathology datasets