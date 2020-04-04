

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            // s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    printf("start make\n");
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    printf("finish num detect\n");
    if(num) *num = nboxes;
    printf("number of box %d\n",nboxes );

    size_t det_size = sizeof(detection);

    printf("size det %d\n",(int)det_size );

    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    
    printf("finish det detect \n");
   
    for(i = 0; i < nboxes; ++i)
    {
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
      
        if(l.coords > 4)
        {
            dets[i].mask = (float*)calloc(l.coords-4, sizeof(float));
        }
    }
    
    printf("finish make\n");
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
           // int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
           // dets += count;
        }
        if(l.type == REGION){
           // get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
           // dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}