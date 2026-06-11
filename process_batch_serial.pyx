# file: process_batch_serial.pyx

import cython
import numpy as np
cimport numpy as cnp
from numpy import argsort

from libc.stdio cimport *
 
cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)

    int fprintf(FILE *stream, const char *format_string, ...)





#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cnp.import_array()

#@cython.wraparound(False)

#@cython.cfunc

#@cython.wraparound(False)
#@cython.boundscheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int, int, int, int, int, int) process_a_fault(int num_fault, int num_batch, int num_classes, int dim_batch, float[:, :] clean_output, double[:,:,:] faulty_output, long[:] batch_labels):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0
    cdef int best5_c = 0
    cdef int masked = 0
    cdef bint masked_flag = 1
    cdef int critical = 0
    cdef int not_critical = 0
    #cdef char faulty_sdc_1 = 'Y'
    #cdef char clean_sdc_1 = 'Y'
    #cdef char best5_flag = 'N'
    #cdef char clean_sdc_5 = 'Y'
    cdef int i = 0
    cdef int j = 0
    cdef int z = 0
    cdef int iterate_over = 0

    cdef long[:] sorted_indices_clean
    cdef long[:] sorted_indices_faulty

    cdef float[:] clean_output_image
    cdef double[:] faulty_output_image
    cdef int clean_output_argmax = 0
    cdef int faulty_output_argmax = 0
    cdef long clean_output_label
    cdef bint faulty_output_match_flag = 0
    cdef bint clean_output_match_flag = 0
    cdef bint best5_c_flag = 0
    cdef bint clean_output_match_top5_flag = 0
    cdef bint faulty_output_match_top5_flag = 0
    cdef int five_indexer = 0
    iterate_over = dim_batch
    #inside batches
    cdef bint DEBUG = 0
    
    while j < iterate_over:

        #Set the flag
        masked_flag = 1
        best5_c_flag = 0
        faulty_output_match_flag = 0
        clean_output_match_flag = 0
        clean_output_match_top5_flag = 0
        faulty_output_match_top5_flag = 0


        clean_output_image = clean_output[j]
        faulty_output_image = faulty_output[num_fault][j]

        sorted_indices_clean = argsort(clean_output_image)
        sorted_indices_faulty = argsort(faulty_output_image)
        
        clean_output_label = batch_labels[j]

        five_indexer = num_classes-1

        clean_output_argmax =  int(sorted_indices_clean[five_indexer])
        faulty_output_argmax = int(sorted_indices_faulty[five_indexer])
        
        
        # Check if correct label is in the top5 clean output (TOP5 Accuracy clean)
        while five_indexer >= num_classes - 5:
            if clean_output_label == sorted_indices_clean[five_indexer]:
                clean_output_match_top5_flag = 1
                #clean_sdc_5 = 'N'
                break
            five_indexer -= 1
                
        if clean_output_match_top5_flag == 1:
            clean_output_match_counter_sdc5 += 1
        
        # Check if correct label is in the top5 faulty output (TOP5 Accuracy faulty)
        five_indexer = num_classes-1
        while five_indexer >= num_classes - 5:
            if clean_output_label == sorted_indices_faulty[five_indexer]:
                faulty_output_match_top5_flag = 1
            if clean_output_argmax == sorted_indices_faulty[five_indexer]:
                best5_c_flag = 1
                #faulty_sdc_5 = 'N'
            five_indexer -= 1
               
        if faulty_output_match_top5_flag == 1:
            faulty_output_match_counter_sdc5 += 1
        if best5_c_flag == 0:
            best5_c += 1
        
        #Clean output match (TOP1 Accuracy clean)
        if clean_output_argmax == clean_output_label:
            clean_output_match_flag = 1
            clean_output_match_counter += 1

        #Faulty output match (TOP1 Accuracy faulty)
        if faulty_output_argmax == clean_output_label:     
            faulty_output_match_flag = 1
            faulty_output_match_counter += 1
        
        # Increment counters based on matches
        #if clean_output_match_flag == 1:
            
            #clean_sdc_1 = 'N'

        #if faulty_output_match_flag == 1:
            #faulty_output_match_counter += 1
            #faulty_sdc_1 = 'N'
        
        #Check if the output is masked
        five_indexer = num_classes-1
        while five_indexer >= 0:
            if clean_output_image[five_indexer] != faulty_output_image[five_indexer]:
                masked_flag = 0
                break
            five_indexer -= 1
        
        if masked_flag == 1:
            masked += 1
            #output_results_list.append('masked')
            #csv_writer.writerow([z, i, j, 'masked', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
        elif clean_output_argmax == faulty_output_argmax:
            not_critical += 1
            #output_results_list.append('not_crit')
            #csv_writer.writerow([z, i, j, 'not_crit', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
        else:
            critical += 1
            #output_results_list.append('SDC-1')
            #csv_writer.writerow([z, i, j, 'SDC-1', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
        j += 1
    return clean_output_match_counter, faulty_output_match_counter, clean_output_match_counter_sdc5, faulty_output_match_counter_sdc5, best5_c, masked, critical, not_critical
    




cpdef (int, int, int, int, int, int, int, int) process_a_fault_writing_given_images(int num_fault, int num_classes, int num_images, long[:] index_list, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char* fname, bint WRITE):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0

    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0
    cdef int i, z, iterate_over
    
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = num_images

    cdef Py_UNICODE clean_sdc_5='Y'
    cdef Py_UNICODE faulty_sdc_5='Y'
    cdef Py_UNICODE clean_sdc_1='Y'
    cdef Py_UNICODE faulty_sdc_1='Y'
    cdef Py_UNICODE best5_flag='Y'

    cdef FILE* output_file = fopen(fname, "a")
    if num_fault == 0:
        if WRITE:
            fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s,%s\n",<char*>'fault', <char*>'image', <char*>'output', <char*>'best 5',<char*> 'clean SDC-1', <char*>'faulty SDC-1', <char*>'clean SDC-5', <char*>'faulty SDC-5')

    #inside batches
    cdef bint DEBUG = False
    j: cython.int  = 0

    while j < iterate_over:
        faulty_sdc_1 = 'Y'
        clean_sdc_1 = 'Y'
        best5_flag = 'Y'
        faulty_sdc_5 = 'Y'
        clean_sdc_5 = 'Y'
        i = index_list[j] 
        masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(num_fault, i, num_classes, clean_output, faulty_output, batch_labels)

        if clean_output_match_top5_flag == 1:
            clean_output_match_counter_sdc5 += 1
            clean_sdc_5 = 'N' 

        
        # Check if correct label is in the top5 faulty output (TOP5 Accuracy faulty)
           
        if faulty_output_match_top5_flag == 1:
            faulty_output_match_counter_sdc5 += 1
            faulty_sdc_5 = 'N'
            
        if best5_c_flag == 0:
            best5_c += 1
            best5_flag = 'N'
        
        #Clean output match (TOP1 Accuracy clean)
        if clean_output_match_flag == 1:
            clean_output_match_counter += 1
            clean_sdc_1 = 'N'
            
        #Faulty output match (TOP1 Accuracy faulty)
        if faulty_output_match_flag == 1:
            faulty_output_match_counter += 1
            faulty_sdc_1 = 'N'
        
        if masked_flag == 1:
            masked += 1
            #output_results_list.append('masked')
            #csv_writer.writerow([z, i, j, 'masked', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, i, <char*>'masked', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)

        elif not_critical_flag == 1:
            not_critical += 1
            #output_results_list.append('not_crit')
            #csv_writer.writerow([z, i, j, 'not_crit', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, <int>i,  <char*>'not_crit', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)
        else:
            critical += 1
            #output_results_list.append('SDC-1')
            #csv_writer.writerow([z, i, j, 'SDC-1', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, <int>i,  <char*>'SDC-1', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)
        j = j + 1
    fclose(output_file)
    return clean_output_match_counter, faulty_output_match_counter, clean_output_match_counter_sdc5, faulty_output_match_counter_sdc5, best5_c, masked, critical, not_critical



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int, int, float, float ) process_a_fault_writing_given_images_and_faults_opt_fi(int num_faults, int num_classes, int num_images, long[:] faults_index_list, long[:] index_list, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char* fname, bint WRITE):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0


    cdef int total = 0
    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0

    cdef float clean_acc = 0
    cdef float faulty_acc = 0

    cdef int z, iterate_over
    cdef long i,n_fault
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = num_images
    cdef bint DEBUG = 1
    cdef FILE* output_file = fopen(fname, "w")
    
    if WRITE:
        fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s\n",<char*>'image_sample_size', <char*>'masked', <char*>'not_critical', <char*>'critical',<char*> 'total', <char*>'clean_acc', <char*>'faulty_acc')

    j: cython.int  = 0
    fault_to_analyze: cython.int = 0
    n_fault = 0
    # Iterate over images
    while j < iterate_over:
        i = index_list[j] 
        fault_to_analyze = 0
        #Iterate over faults
        while fault_to_analyze < num_faults:
            n_fault = faults_index_list[fault_to_analyze]
            masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(n_fault, i, num_classes, clean_output, faulty_output, batch_labels)

            #Clean output match (TOP1 Accuracy clean)
            if clean_output_match_flag == 1:
                clean_output_match_counter += 1
                
            #Faulty output match (TOP1 Accuracy faulty)
            if faulty_output_match_flag == 1:
                faulty_output_match_counter += 1
            
            if masked_flag == 1:
                masked += 1
            elif not_critical_flag == 1:
                not_critical += 1
            else:
                critical += 1
            fault_to_analyze = fault_to_analyze + 1 
        
       
        j = j + 1
        if j % 1000 == 0:
            printf("Processed %d images\n",j)
        total = masked + not_critical + critical
        clean_acc = <float>(100*<float>clean_output_match_counter / (j*(num_faults+1)))
        faulty_acc = <float>(100*<float>faulty_output_match_counter / (j*(num_faults+1)))
        if WRITE:
            fprintf(output_file,"%d,%d,%d,%d,%d,%f,%f\n",j, masked, not_critical, critical, total, clean_acc,faulty_acc)

    fclose(output_file)
    return masked, not_critical,critical, total, clean_acc, faulty_acc

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int, int, float, float ) process_a_fault_writing_given_images_and_faults_opt_fi_plus_detection(int num_faults, int num_classes, int num_images, long[:] faults_index_list, long[:] index_list, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char[:] detected, char* fname, char* fname_images, bint WRITE, bint WRITE_IMAGES):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0


    cdef int total = 0
    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0

    cdef float clean_acc = 0
    cdef float faulty_acc = 0

    cdef int z, iterate_over
    cdef long i,n_fault
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = num_images
    cdef bint DEBUG = 1
    cdef FILE* output_file 
    if WRITE:
        output_file = fopen(fname, "w")
    cdef FILE* image_output_file
    if WRITE_IMAGES:
        image_output_file =  fopen(fname_images, "w")

    if WRITE:
        fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",<char*>'image_sample_size', <char*>'masked', <char*>'not_critical', <char*>'critical',<char*> 'total', <char*>'clean_acc', <char*>'faulty_acc',<char*>'masked_f',<char*>'not_critical_f',<char*>'critical_f')
    if WRITE_IMAGES:
        fprintf(image_output_file,"%s,%s,%s,%s,%s\n",<char*>'image', <char*>'fault', <char*>'fault_outcome', <char*>'clean_matched',<char*> 'faulty_matched')

    cdef char fault_outcome = 0
    j: cython.int  = 0
    fault_to_analyze: cython.int = 0
    n_fault = 0
    # Iterate over images
    cdef int masked_f,not_critical_f,critical_f
    masked_f = 0
    not_critical_f = 0
    critical_f = 0
    while j < iterate_over:
        i = index_list[j] 
        fault_to_analyze = 0
        #Iterate over faults
        
        while fault_to_analyze < num_faults:
            n_fault = faults_index_list[fault_to_analyze]
            masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(n_fault, i, num_classes, clean_output, faulty_output, batch_labels)

            #Clean output match (TOP1 Accuracy clean)
            if clean_output_match_flag == 1:
                clean_output_match_counter += 1
                
            #Faulty output match (TOP1 Accuracy faulty)
            if faulty_output_match_flag == 1:
                faulty_output_match_counter += 1
            
            if masked_flag == 1:
                masked += 1
            elif not_critical_flag == 1:
                not_critical += 1
            else:
                critical += 1
            
            fault_to_analyze = fault_to_analyze + 1 

            fault_outcome = 0 if masked_flag == 1 else 1 if not_critical_flag == 1 else 2
            if detected[n_fault] != fault_outcome:
                if detected[n_fault] == 0 or detected[n_fault] == -1 or (detected[n_fault] == 1 and fault_outcome == 2):
                    detected[n_fault] = fault_outcome
            if WRITE_IMAGES:
                fprintf(image_output_file,"%ld,%ld,%d,%u,%u\n",index_list[j],n_fault,fault_outcome, clean_output_match_flag, faulty_output_match_flag)
       

        fault_to_analyze = 0
        #Iterate over faults
        masked_f = 0
        not_critical_f = 0
        critical_f = 0
        while fault_to_analyze < num_faults:
            n_fault = faults_index_list[fault_to_analyze]
            fault_outcome = detected[n_fault]
            if fault_outcome == 0:
                masked_f += 1
            elif fault_outcome == 1:
                not_critical_f += 1
            elif fault_outcome == 2:
                critical_f += 1
            fault_to_analyze = fault_to_analyze + 1

        j = j + 1
        if j % 1000 == 0:
            printf("Processed %d images\n",j)
        total = masked + not_critical + critical
        clean_acc = <float>(100*<float>clean_output_match_counter / (j*(num_faults+1)))
        faulty_acc = <float>(100*<float>faulty_output_match_counter / (j*(num_faults+1)))
        if WRITE:
            fprintf(output_file,"%d,%d,%d,%d,%d,%f,%f,%d,%d,%d\n",j, masked, not_critical, critical, total, clean_acc,faulty_acc,masked_f,not_critical_f,critical_f)

    if WRITE_IMAGES:
        fclose(image_output_file)
    if WRITE:
        fclose(output_file)
    return masked, not_critical,critical, total, clean_acc, faulty_acc

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.exceptval(check=False)
cpdef (int, int, int, int, float, float ) process_a_fault_writing_given_images_and_faults_opt_fi_plus_detection_incremental(int num_faults, int num_classes, int num_images, long[:] faults_index_list, long[:] index_list, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char[:] detected, char* fname, char* fname_images, bint WRITE, bint WRITE_IMAGES, int writing_from):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0


    cdef int total = 0
    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0

    cdef float clean_acc = 0
    cdef float faulty_acc = 0

    cdef int z, iterate_over
    cdef long i,n_fault
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = num_images
    cdef bint DEBUG = 1
    cdef FILE* output_file 
    if WRITE:
        output_file = fopen(fname, "w")
    cdef FILE* image_output_file


    if WRITE_IMAGES:
        if writing_from == 0:
            image_output_file =  fopen(fname_images, "w")
        else:
            image_output_file =  fopen(fname_images, "a")

    if WRITE:
        if writing_from == 0:
            fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",<char*>'image_sample_size', <char*>'masked', <char*>'not_critical', <char*>'critical',<char*> 'total', <char*>'clean_acc', <char*>'faulty_acc',<char*>'masked_f',<char*>'not_critical_f',<char*>'critical_f')
    if WRITE_IMAGES:
        if writing_from == 0:
            fprintf(image_output_file,"%s,%s,%s,%s,%s\n",<char*>'image', <char*>'fault', <char*>'fault_outcome', <char*>'clean_matched',<char*> 'faulty_matched')

    cdef char fault_outcome = 0
    j: cython.int  = 0
    fault_to_analyze: cython.int = 0
    n_fault = 0
    # Iterate over images
    cdef int masked_f,not_critical_f,critical_f
    masked_f = 0
    not_critical_f = 0
    critical_f = 0
    while j < iterate_over:
        i = index_list[j] 
        fault_to_analyze = 0
        #Iterate over faults
        
        while fault_to_analyze < num_faults:
            n_fault = faults_index_list[fault_to_analyze]
            masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(n_fault, i, num_classes, clean_output, faulty_output, batch_labels)

            #Clean output match (TOP1 Accuracy clean)
            if clean_output_match_flag == 1:
                clean_output_match_counter += 1
                
            #Faulty output match (TOP1 Accuracy faulty)
            if faulty_output_match_flag == 1:
                faulty_output_match_counter += 1
            
            if masked_flag == 1:
                masked += 1
            elif not_critical_flag == 1:
                not_critical += 1
            else:
                critical += 1
            
            fault_to_analyze = fault_to_analyze + 1 

            fault_outcome = 0 if masked_flag == 1 else 1 if not_critical_flag == 1 else 2
            if detected[n_fault] != fault_outcome:
                if detected[n_fault] == 0 or detected[n_fault] == -1 or (detected[n_fault] == 1 and fault_outcome == 2):
                    detected[n_fault] = fault_outcome
            if WRITE_IMAGES:
                fprintf(image_output_file,"%ld,%ld,%d,%u,%u\n",writing_from + index_list[j],n_fault,fault_outcome, clean_output_match_flag, faulty_output_match_flag)
       

        fault_to_analyze = 0
        #Iterate over faults
        masked_f = 0
        not_critical_f = 0
        critical_f = 0
        while fault_to_analyze < num_faults:
            n_fault = faults_index_list[fault_to_analyze]
            fault_outcome = detected[n_fault]
            if fault_outcome == 0:
                masked_f += 1
            elif fault_outcome == 1:
                not_critical_f += 1
            elif fault_outcome == 2:
                critical_f += 1
            fault_to_analyze = fault_to_analyze + 1

        j = j + 1
        if j % 512 == 0:
            printf("Processed %d images\n",writing_from + j)
        total = masked + not_critical + critical
        clean_acc = <float>(100*<float>clean_output_match_counter / (j*(num_faults+1)))
        faulty_acc = <float>(100*<float>faulty_output_match_counter / (j*(num_faults+1)))
        if WRITE:
            fprintf(output_file,"%d,%d,%d,%d,%d,%f,%f,%d,%d,%d\n",writing_from + j, masked, not_critical, critical, total, clean_acc,faulty_acc,masked_f,not_critical_f,critical_f)

    if WRITE_IMAGES:
        fclose(image_output_file)
    if WRITE:
        fclose(output_file)
    return masked, not_critical,critical, total, clean_acc, faulty_acc

cpdef (int, int, int, int, float, float ) process_a_fault_writing_given_images_opt_fi(int num_faults, int num_classes, int num_images, long[:] index_list, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char* fname, char* fname_images, bint WRITE, bint WRITE_IMAGES):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0


    cdef int total = 0

    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0

    cdef float clean_acc = 0
    cdef float faulty_acc = 0

    cdef int fault_outcome 
    cdef int z, iterate_over
    cdef long i
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = num_images


    cdef FILE* output_file = fopen(fname, "w")
    
    cdef FILE* images_output_file
    
    if WRITE_IMAGES:
        images_output_file = fopen(fname_images, "w")

    if WRITE:
        fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s\n",<char*>'image_sample_size', <char*>'masked', <char*>'not_critical', <char*>'critical',<char*> 'total', <char*>'clean_acc', <char*>'faulty_acc')

    if WRITE_IMAGES:
        fprintf(output_file,"%s,%s,%s,%s,%s\n",<char*>'image', <char*>'fault', <char*>'fault_outcome', <char*>'clean_matched',<char*> 'faulty_matched')

    cdef bint DEBUG = False
    j: cython.int  = 0
    fault_to_analyze: cython.int = 0
    # Iterate over images
    while j < iterate_over:
        i = index_list[j] 
        fault_to_analyze = 0
        #Iterate over faults
        while fault_to_analyze < num_faults:
            masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(fault_to_analyze, i, num_classes, clean_output, faulty_output, batch_labels)

            #Clean output match (TOP1 Accuracy clean)
            if clean_output_match_flag == 1:
                clean_output_match_counter += 1
                
            #Faulty output match (TOP1 Accuracy faulty)
            if faulty_output_match_flag == 1:
                faulty_output_match_counter += 1
            
            if masked_flag == 1:
                masked += 1
            elif not_critical_flag == 1:
                not_critical += 1
            else:
                critical += 1
            fault_to_analyze = fault_to_analyze + 1 

            fault_outcome = 0 if masked_flag == 1 else 1 if not_critical_flag == 1 else 2
            if WRITE_IMAGES:
                fprintf(images_output_file,"%ld,%d,%d,%c,%c\n",i,fault_to_analyze,fault_outcome, clean_output_match_flag, faulty_output_match_flag)
       
        j = j + 1
        if j % 1000 == 0:
            printf("Processed %d images\n",j)
        total = masked + not_critical + critical
        clean_acc = <float>(100*<float>clean_output_match_counter / (j*(num_faults+1)))
        faulty_acc = <float>(100*<float>faulty_output_match_counter / (j*(num_faults+1)))
        if WRITE:
            fprintf(output_file,"%d,%d,%d,%d,%d,%f,%f\n",j, masked, not_critical, critical, total, clean_acc,faulty_acc)

    fclose(output_file)
    return masked, not_critical,critical, total, clean_acc, faulty_acc

cpdef (int, int, int, int, int, int, int, int) process_a_fault_writing(int num_fault, int num_batch, int num_classes, int dim_batch, int start_from, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char* fname, bint WRITE):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0

    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0
    cdef int i, iterate_over
    cdef char fault_outcome = 0
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = dim_batch

   
    cdef FILE* output_file = fopen(fname, "a")

    if num_fault == 0 and num_batch == 0:
        if WRITE:
            fprintf(output_file,"%s,%s,%s,%s,%s\n",<char*>'image', <char*>'fault', <char*>'fault_outcome', <char*>'clean_matched',<char*> 'faulty_matched')
    #inside batches
    cdef bint DEBUG = False
    j: cython.int  = 0

    while j < iterate_over:
        masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(num_fault, j, num_classes, clean_output, faulty_output, batch_labels)
        fault_outcome = 0 if masked_flag == 1 else 1 if not_critical_flag == 1 else 2
        if WRITE:
            fprintf(output_file,"%ld,%ld,%u,%u,%u\n",start_from + j,num_fault,fault_outcome, clean_output_match_flag, faulty_output_match_flag)
        j = j+ 1
        if fault_outcome == 0:
            masked+=1
        elif fault_outcome == 1:
            not_critical+=1
        else:
            critical+=1
        
        clean_output_match_counter += clean_output_match_flag
        faulty_output_match_counter += faulty_output_match_flag
        clean_output_match_counter_sdc5 += clean_output_match_top5_flag
        faulty_output_match_counter_sdc5 += faulty_output_match_top5_flag
        best5_c += best5_c_flag

    fclose(output_file)
    return clean_output_match_counter, faulty_output_match_counter, clean_output_match_counter_sdc5, faulty_output_match_counter_sdc5, best5_c, masked, critical, not_critical


cpdef (int, int, int, int, int, int, int, int) process_a_fault_writing_(int num_fault, int num_batch, int num_classes, int dim_batch, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels, char* fname, bint WRITE):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0

    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0
    cdef int i, z, iterate_over
    cdef char fault_outcome = 0 
    cdef bint masked_flag,not_critical_flag, faulty_output_match_flag, clean_output_match_flag, best5_c_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    iterate_over = dim_batch

    cdef Py_UNICODE clean_sdc_5='Y'
    cdef Py_UNICODE faulty_sdc_5='Y'
    cdef Py_UNICODE clean_sdc_1='Y'
    cdef Py_UNICODE faulty_sdc_1='Y'
    cdef Py_UNICODE best5_flag='Y'
    cdef FILE* output_file = fopen(fname, "a")
    if num_fault == 0 and num_batch == 0:
        if WRITE:
            fprintf(output_file,"%s,%s,%s,%s,%s,%s,%s,%s,%s\n",<char*>'fault', <char*>'batch', <char*>'image', <char*>'output', <char*>'best 5',<char*> 'clean SDC-1', <char*>'faulty SDC-1', <char*>'clean SDC-5', <char*>'faulty SDC-5')

    #inside batches
    cdef bint DEBUG = False
    j: cython.int  = 0

    while j < iterate_over:
        faulty_sdc_1 = 'Y'
        clean_sdc_1 = 'Y'
        best5_flag = 'Y'
        faulty_sdc_5 = 'Y'
        clean_sdc_5 = 'Y'
        masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag = analyze_a_fault(num_fault, j, num_classes, clean_output, faulty_output, batch_labels)

        if clean_output_match_top5_flag == 1:
            clean_output_match_counter_sdc5 += 1
            clean_sdc_5 = 'N' 

        
        # Check if correct label is in the top5 faulty output (TOP5 Accuracy faulty)
           
        if faulty_output_match_top5_flag == 1:
            faulty_output_match_counter_sdc5 += 1
            faulty_sdc_5 = 'N'
            
        if best5_c_flag == 0:
            best5_c += 1
            best5_flag = 'N'
        
        #Clean output match (TOP1 Accuracy clean)
        if clean_output_match_flag == 1:
            clean_output_match_counter += 1
            clean_sdc_1 = 'N'
            
        #Faulty output match (TOP1 Accuracy faulty)
        if faulty_output_match_flag == 1:
            faulty_output_match_counter += 1
            faulty_sdc_1 = 'N'
        
        if masked_flag == 1:
            masked += 1
            #output_results_list.append('masked')
            #csv_writer.writerow([z, i, j, 'masked', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, num_batch, <int>j, <char*>'masked', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)

        elif not_critical_flag == 1:
            not_critical += 1
            #output_results_list.append('not_crit')
            #csv_writer.writerow([z, i, j, 'not_crit', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, num_batch, <int>j,  <char*>'not_crit', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)
        else:
            critical += 1
            #output_results_list.append('SDC-1')
            #csv_writer.writerow([z, i, j, 'SDC-1', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5])
            if WRITE:
                fprintf(output_file,"%d,%d,%d,%s,%c,%c,%c,%c,%c\n",num_fault, num_batch, <int>j,  <char*>'SDC-1', best5_flag, clean_sdc_1, faulty_sdc_1, clean_sdc_5, faulty_sdc_5)
        j = j+ 1
    fclose(output_file)
    return clean_output_match_counter, faulty_output_match_counter, clean_output_match_counter_sdc5, faulty_output_match_counter_sdc5, best5_c, masked, critical, not_critical

cdef (bint, bint, bint, bint, bint, bint, bint ) analyze_a_fault (int num_fault, int num_image, int num_classes, float[:, :] clean_output, float[:,:,:] faulty_output, long[:] batch_labels):
    #Set the flag
    cdef bint masked_flag = 1
    cdef bint not_critical_flag = 0
    cdef bint best5_c_flag = 0
    cdef bint faulty_output_match_flag = 0
    cdef bint clean_output_match_flag = 0
    cdef bint clean_output_match_top5_flag = 0
    cdef bint faulty_output_match_top5_flag = 0

    cdef long[:] sorted_indices_clean
    cdef long[:] sorted_indices_faulty
    cdef float[:] clean_output_image
    cdef float[:] faulty_output_image

    cdef int five_indexer

    cdef int clean_output_argmax = 0
    cdef int faulty_output_argmax = 0

    cdef long clean_output_label
    clean_output_image = clean_output[num_image]
    faulty_output_image = faulty_output[num_fault][num_image]

    sorted_indices_clean = argsort(clean_output_image)
    sorted_indices_faulty = argsort(faulty_output_image)
    
    clean_output_label = batch_labels[num_image]

    five_indexer = num_classes-1

    clean_output_argmax = <int>sorted_indices_clean[five_indexer]
    faulty_output_argmax = <int>sorted_indices_faulty[five_indexer]
    
    
    # Check if correct label is in the top5 clean output (TOP5 Accuracy clean)
    while five_indexer >= num_classes - 5:
        if clean_output_label == sorted_indices_clean[five_indexer]:
            clean_output_match_top5_flag = 1
            #clean_sdc_5 = 'N'
            break
        five_indexer -= 1
            
    
    # Check if correct label is in the top5 faulty output (TOP5 Accuracy faulty)
    five_indexer = num_classes-1
    while five_indexer >= num_classes - 5:
        if clean_output_label == sorted_indices_faulty[five_indexer]:
            faulty_output_match_top5_flag = 1
        if clean_output_argmax == sorted_indices_faulty[five_indexer]:
            best5_c_flag = 1
            
        five_indexer -= 1

    #Check if the output is masked
    five_indexer = num_classes-1
    while five_indexer >= 0:
        if clean_output_image[five_indexer] != faulty_output_image[five_indexer]:
            masked_flag = 0
            break
        five_indexer -= 1
    
    if clean_output_argmax == faulty_output_argmax:
        not_critical_flag = 1

    #Clean output match (TOP1 Accuracy clean)
    if clean_output_argmax == clean_output_label:
        clean_output_match_flag = 1
    #Faulty output match (TOP1 Accuracy faulty)
    if faulty_output_argmax == clean_output_label:     
        faulty_output_match_flag = 1

    return masked_flag, not_critical_flag, best5_c_flag, faulty_output_match_flag, clean_output_match_flag, clean_output_match_top5_flag, faulty_output_match_top5_flag
    
cpdef (int, int, int, int, int, int, int, int) process_a_batch(int num_faults, int num_batch, int num_classes, int dim_batch, float[:, :] clean_output, double[:,:,:] faulty_output, long[:] batch_labels):
    cdef int clean_output_match_counter = 0
    cdef int faulty_output_match_counter = 0
    cdef int clean_output_match_counter_sdc5 = 0
    cdef int faulty_output_match_counter_sdc5 = 0
    cdef int best5_c = 0
    cdef int masked = 0
    cdef int critical = 0
    cdef int not_critical = 0
    
    cdef int z = 0

    cdef int clean_output_match_counter_local = 0
    cdef int faulty_output_match_counter_local = 0
    cdef int clean_output_match_counter_sdc5_local = 0
    cdef int faulty_output_match_counter_sdc5_local = 0
    cdef int best5_c_local = 0
    cdef int masked_local = 0
    cdef int critical_local = 0
    cdef int not_critical_local = 0
        
    #inside batches
    while z < num_faults:
        clean_output_match_counter_local = 0
        faulty_output_match_counter_local = 0
        clean_output_match_counter_sdc5_local = 0
        faulty_output_match_counter_sdc5_local = 0
        best5_c_local = 0
        masked_local = 0
        critical_local = 0
        not_critical_local = 0

        clean_output_match_counter_local, faulty_output_match_counter_local, clean_output_match_counter_sdc5_local, faulty_output_match_counter_sdc5_local, best5_c_local, masked_local, critical_local, not_critical_local = process_a_fault(z, num_batch, num_classes, dim_batch, clean_output, faulty_output, batch_labels)
        
        clean_output_match_counter += clean_output_match_counter_local
        faulty_output_match_counter += faulty_output_match_counter_local
        clean_output_match_counter_sdc5 += clean_output_match_counter_sdc5_local
        faulty_output_match_counter_sdc5 += faulty_output_match_counter_sdc5_local
        best5_c += best5_c_local
        masked += masked_local
        critical += critical_local
        not_critical += not_critical_local

        z = z + 1
    
    return clean_output_match_counter, faulty_output_match_counter, clean_output_match_counter_sdc5, faulty_output_match_counter_sdc5, best5_c, masked, critical, not_critical
    


'''
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
 
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    np.int_ index
    np.float32 value

cdef int _compare(const_void *a, const_void *b):
    cdef np.float32 v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1

cdef argsort(np.float32_t[:] data, np.intp_t[:] order) nogil:
    cdef np.int_ i
    cdef np.int_ n = data.shape[0]
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)
'''