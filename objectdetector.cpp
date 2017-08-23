#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif



#if defined(__cplusplus)
extern "C" {
#endif

#include "darknet.h"

#if defined(__cplusplus)
}
#endif

#include "opencv2/opencv.hpp"
#include <apps/Common/exampleHelper.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire_GenICam.h>


using namespace mvIMPACT::acquire;
using namespace cv;


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
        list *options = read_data_cfg(datacfg);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);

        image **alphabet = load_alphabet();
        network net = parse_network_cfg(cfgfile);
        if(weightfile) {
                load_weights(&net, weightfile);
        }
        set_batch_network(&net, 1);


        srand(2222222);
        double time;
        char buff[256];
        char *input = buff;
        int j;
        float nms=.3;
        while(1) {
                if(filename) {
                        strncpy(input, filename, 256);
                } else {
                        printf("Enter Image Path: ");
                        fflush(stdout);
                        input = fgets(input, 256, stdin);
                        if(!input) return;
                        strtok(input, "\n");
                }
                image im = load_image_color(input,0,0);
                printf("im : %d %d %d \n", im.w, im.h, im.c);
                image sized = letterbox_image(im, net.w, net.h);
                printf("sized im : %d %d %d \n", sized.w, sized.h, sized.c);
                //image sized = resize_image(im, net.w, net.h);
                //image sized2 = resize_max(im, net.w);
                //image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
                //resize_network(&net, sized.w, sized.h);
                layer l = net.layers[net.n-1];

                box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
                float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
                for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
                float **masks = 0;
                if (l.coords > 4) {
                        masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
                        for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float*)calloc(l.coords-4, sizeof(float *));
                }

                float *X = sized.data;
                time=what_time_is_it_now();
                network_predict(net, X);
                printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
                get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
                if (nms) {
                        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
                }
                //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
                draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
                if(outfile) {
                        save_image(im, outfile);
                }
                else{
                        save_image(im, "predictions");
#ifdef OPENCV
                        cvNamedWindow("predictions", CV_WINDOW_NORMAL);
                        if(fullscreen) {
                                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                        }
                        show_image(im, "predictions");
                        cvWaitKey(0);
                        cvDestroyAllWindows();
#endif
                }

                free_image(im);
                free_image(sized);
                free(boxes);
                free_ptrs((void **)probs, l.w*l.h*l.n);
                if (filename) break;
        }
}


int setting(Device* dev)
{
        const int ROI_WIDTH = 1024; //640;
        const int ROI_HEIGHT = 768; //480;
        // MAx : 1936 X 1216
        //ROI 설정하는 부분
        mvIMPACT::acquire::GenICam::ImageFormatControl ifct(dev);
        ifct.width.write(ROI_WIDTH);
        ifct.height.write(ROI_HEIGHT);
        ifct.offsetX.write(968-ROI_WIDTH/2.0);
        ifct.offsetY.write(608-ROI_HEIGHT/2.0);

        //Gain 설정하는 부분
        mvIMPACT::acquire::GenICam::AnalogControl aGainCtrl(dev);
        aGainCtrl.gain.writeS("2.640");

        //Exposure 설정하는 부분 (us)
        mvIMPACT::acquire::GenICam::AcquisitionControl acqCtrl(dev);
        acqCtrl.exposureTime.writeS("6000");

        return 0;
}

int main(int argc, char **argv)
{

        // char *outfile = find_char_arg(argc, argv, "-out", 0);
        // int fullscreen = find_arg(argc, argv, "-fullscreen");
        // test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);

        DeviceManager devMgr;
        if( devMgr.deviceCount() == 0 ) {
                std::cout << "No device found! Unable to continue!" << std::endl;
                return -1;
        }

        Device* pDev = devMgr[0];

        if ( pDev ) {
                conditionalSetProperty(pDev->interfaceLayout, mvIMPACT::acquire::dilGenICam, true);
        }
        else {
                return -1;
        }

        try
        {
                pDev->open();
        }
        catch( const ImpactAcquireException& e )
        {
                // this e.g. might happen if the same device is already opened in another process...
                std::cout << "An error occurred while opening the device " << pDev->serial.read() << " (error code: " << e.getErrorCode() << "). Press any key to end the application..." << std::endl;
                return -2;
        }

        std::cout << "The device " << pDev->serial.read() << " has been opened." << std::endl;
        // if ( setting(pDev) < 0 ) {
        //         std::cout << "ERROR Setting" << std::endl;
        // }


        std::cout << "The device " << pDev->serial.read() << " has been opened." << std::endl;
        FunctionInterface fi( pDev );


        int result = DMR_NO_ERROR;
        SystemSettings ss( pDev );
        const int REQUEST_COUNT = ss.requestCount.read();
        for( int i=0; i<REQUEST_COUNT; i++ )
        {
                result = fi.imageRequestSingle();
                if( result != DMR_NO_ERROR )
                {
                        std::cout << "Error while filling the request queue: " << ImpactAcquireException::getErrorCodeAsString( result ) << std::endl;
                }
        }

        const Request* pRequest = 0;
        int requestNr = -1;
        Mat img;


        //namedWindow("edges",1);


        ////////////////////////////////////////////////////////////////////////
        // init CNN
        float thresh = 0.24f;
        float hier_thresh = 0.5f;
        char* datacfg = "cfg/coco.data";
        char *cfgfile = "cfg/yolo.cfg";
        char *weightfile = "darknet/yolo.weights";

        list *options = read_data_cfg(datacfg);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);

        image **alphabet = load_alphabet();
        network net = parse_network_cfg(cfgfile);
        if(weightfile) {
                load_weights(&net, weightfile);
        }
        set_batch_network(&net, 1);

        srand(2222222);
        double time;
        char buff[256];
        char *input = buff;
        int j;
        float nms = .3;

        while( true ) {
                requestNr = fi.imageRequestWaitFor( 30 );
                if( fi.isRequestNrValid( requestNr ) ) {
                        pRequest = fi.getRequest(requestNr);
                        if( pRequest->isOK() ) {
                                // display/process/store or do whatever you like with the image here

				/*
                                img = Mat(
                                        pRequest->imageHeight.read(),
                                        pRequest->imageWidth.read(),
                                        CV_8UC1,
                                        pRequest->imageData.read());
				*/

                                // image im = load_image_color(input,0,0)
                                image im;
                                im.w = pRequest->imageWidth.read();
                                im.h = pRequest->imageHeight.read();
                                im.c = 3;
                                int im_size = im.w * im.h;
                                im.data =new float[im_size * im.c];

                                unsigned char* data = (unsigned char*)pRequest->imageData.read();
                                for( int i = 0; i < im.h; ++i ) {
                                        for(int j = 0; j < im.w; ++j) {
                                                im.data[i*im.w + j + im_size*0] = data[i*im.w + j] / 255.0f;
                                                im.data[i*im.w + j + im_size*1] = data[i*im.w + j] / 255.0f;
                                                im.data[i*im.w + j + im_size*2] = data[i*im.w + j] / 255.0f;

                                                // im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
                                        }
                                }

                                image sized = letterbox_image(im, net.w, net.h);
                                layer l = net.layers[net.n-1];

                                box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
                                float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
                                for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
                                float **masks = 0;
                                if (l.coords > 4) {
                                        masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
                                        for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float*)calloc(l.coords-4, sizeof(float *));
                                }

                                float *X = sized.data;
                                time=what_time_is_it_now();
                                network_predict(net, X);
                                printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
                                get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
                                if (nms) {
                                        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
                                }

                                draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
                                
				//save_image(im, "test" );

                                //imshow("edges", img);


                                delete im.data;

                        }
                        else {
                                std::cout << "Error: Request result " << pRequest->requestResult.readS() << std::endl;
                        }

                        fi.imageRequestUnlock( requestNr );
                        fi.imageRequestSingle();
                }
                else {
                        //cout << "Error: There has been no request in the queue!" << endl;
                }

                if(waitKey(1) >= 0) break;
        }

        fi.imageRequestReset( 0, 0 );

        //test_detector("cfg/coco.data", "cfg/yolo.cfg", "yolo.weights", "data/test.png", thresh, .5, outfile, fullscreen);

        return 0;
}
