
<<<<<<< HEAD
using namespace std;

// create darknet 19 (19 cv)
 tensor_t<float> dataX( 256, 256, 3 ) ;

    conv_layer * c1      = new conv_layer(3, 32, 1, 1, dataX.size,1); //Wsize, Nfilter, stride, pad, data.size, activate = 1 :leaky, 0: linear
// pooling 1 2x2, stride 2 // max :0, average:1
	pool_layer * p1 	 = new pool_layer(2, 2, 0, c1->output.size );
	conv_layer * c2      = new conv_layer(3, 64, 1, 1, p1->output.size, 1);
	pool_layer * p2 	 = new pool_layer(2, 2, 0, c2->output.size );
	
	conv_layer * c3      = new conv_layer(3, 128, 1, 1, p2->output.size, 1);
	conv_layer * c4      = new conv_layer(1, 64, 1, 0, c3->output.size,  1); // pad
	conv_layer * c5      = new conv_layer(3, 128, 1, 1, c4->output.size, 1);
	pool_layer * p3 	 = new pool_layer(2, 2, 0, c5->output.size);
	
	conv_layer * c6      = new conv_layer(3, 256, 1, 1, p3->output.size, 1);
	conv_layer * c7      = new conv_layer(1, 128, 1, 0, c6->output.size, 1);
	conv_layer * c8      = new conv_layer(3, 256, 1, 1, c7->output.size, 1);
	pool_layer * p4 	 = new pool_layer(2, 2, 0, c8->output.size );
    
    conv_layer * c9      = new conv_layer(3, 512, 1, 1, p4->output.size, 1);
    conv_layer * c10      = new conv_layer(1, 256, 1, 0, c9->output.size,1);
	conv_layer * c11      = new conv_layer(3, 512, 1, 1, c10->output.size,1);
	conv_layer * c12      = new conv_layer(1, 256, 1, 0, c11->output.size,1);
	conv_layer * c13      = new conv_layer(3, 512, 1, 1, c12->output.size,1);
	pool_layer * p5 	 = new pool_layer(2, 2, 0, c13->output.size );

	conv_layer * c14      = new conv_layer(3, 1024, 1, 1, p5->output.size,1);
	conv_layer * c15      = new conv_layer(1, 512, 1, 0, c14->output.size,1);  
	conv_layer * c16      = new conv_layer(3, 1024, 1, 1, c15->output.size,1);  
	conv_layer * c17      = new conv_layer(1, 512, 1, 0, c16->output.size,1);
	conv_layer * c18      = new conv_layer(3, 1024, 1, 1, c17->output.size,1);
	conv_layer * c19      = new conv_layer(1, 1000, 1, 0, c18->output.size,0);
	pool_layer * p6 	 = new pool_layer(8, 1, 1, c19->output.size );

	classifier * sf_estimate = new classifier(p6->output.size);
/*
	vector<layer*> layers;
	layers.push_back( (layer*) c1 );
	layers.push_back( (layer*) p1);
	layers.push_back( (layer*) c2 );
	layers.push_back( (layer*) p2);

	layers.push_back( (layer*) c3 );
	layers.push_back( (layer*) c4 );
	layers.push_back( (layer*) c5 );
	layers.push_back( (layer*) p3 );

	layers.push_back( (layer*) c6 );
	layers.push_back( (layer*) c7);
	layers.push_back( (layer*) c8 );
	layers.push_back( (layer*) p4);

	layers.push_back( (layer*) c9 );
	layers.push_back( (layer*) c10);
	layers.push_back( (layer*) c11 );
	layers.push_back( (layer*) c12);
	layers.push_back( (layer*) c13);
	layers.push_back( (layer*) p5);

	layers.push_back( (layer*) c14);
	layers.push_back( (layer*) c15);
	layers.push_back( (layer*) c16);
	layers.push_back( (layer*) c17);
	layers.push_back( (layer*) c18);
	layers.push_back( (layer*) c19);
	layers.push_back( (layer*) p6);
	layers.push_back( (layer*) sf_estimate);
*/
=======
// create darknet 19 (19 cv)
 tensor_t<float> dataX( 256, 256, 3 ) ;
// layer conv1 5x5 , 6 filters, stride 1, pad 0
    conv_layer * c1      = new conv_layer(3, 32, 1, 1, dataX.size); //Wsize, Nfilter, stride, pad, dataset[0].data.size
    relu_layer * c1_relu = new relu_layer(c1->output.size);

    
>>>>>>> baa9642075b1cc89267b381b700da28b24a21401
