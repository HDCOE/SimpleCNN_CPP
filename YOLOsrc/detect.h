
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    data_t *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            data_t scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow((float)predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow((float)predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale * predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
                //printf("dets %d prob %f\n",index, (float)dets[index].prob[j] );
            }
        }
    }
}
void forward_detection_layer(layer l, network net)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(data_t));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1, l.output + index + offset);
            }
        }
    }
 }

layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
	layer l;
    memset(&l,0,sizeof(layer));
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;

    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = (data_t*)calloc(1, sizeof(data_t));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = (data_t*)calloc(batch*l.outputs, sizeof(data_t));
  

    l.forward = forward_detection_layer;


    fprintf(stderr, "Detection Layer\n");
    srand(0);

///// return param
    l.params.h = l.out_h;
    l.params.w = l.out_w;
    l.params.c = l.out_c;
    l.params.inputs = l.outputs;
    l.params.batch = batch;
    
    return l;
}

layer parse_detection(int coords, int classes, int side, size_params params)
{

	int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    int softmax = 0;
    int num = 2;
    int rescore = 1;
    int sqrt = 1;

    layer l = make_detection_layer(batch, params.inputs , num, side, classes, coords,rescore);

    l.softmax = 0;
    l.sqrt = 1;
    l.max_boxes = 90;
    l.coord_scale = 5;
    l.forced = 0;
    l.object_scale = 1;
    l.noobject_scale = 0.5;
    l.class_scale = 1;
    l.jitter = 0.2;
    l.random = 0;
    l.reorg = 0;

    return l;
}