// cython methods to speed-up evaluation

void addToConfusionMatrix( const unsigned char* f_prediction_p  ,
                           const unsigned char* f_groundTruth_p ,
                           const unsigned int   f_width_i       ,
                           const unsigned int   f_height_i      ,
                           unsigned long long*  f_confMatrix_p  ,
                           const unsigned int   f_confMatDim_i  )
{
    const unsigned int size_ui = f_height_i * f_width_i;
    for (unsigned int i = 0; i < size_ui; ++i)
    {
        const unsigned char predPx = f_prediction_p [i];
        const unsigned char gtPx   = f_groundTruth_p[i];
        f_confMatrix_p[f_confMatDim_i*gtPx + predPx] += 1u;
    }
}