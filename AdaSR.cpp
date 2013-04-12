#include "AdaSR.h"

AdaSR::AdaSR(QRect view_rect, Object *parent) :
    QObject(parent)
{
    //All tracking will be performed in relation to this view_rect.
    _view_rect = view_rect;
    _initialize = false;
    _tracking = false;
}

void AdaSR::run()
{
    //If initialization is requested then initialize
    if ( _initialize ) {
        qDebug() << "Start Initialize";

        if ( constructSampleWindow() ) {

            //Calculate the integral image HC and HOG
            //Only calculate this for the sample region for speed
//            qDebug() << "Start Frame Integral Calculation: " << QTime::currentTime().toString("hh:mm:ss.zzz");
            cv::Mat frame_hc, frame_hog;
            cv::Mat sample_win = _original_img(cv::Rect(_sample_win.topLeft().x(), _sample_win.topLeft().y(), _sample_win.width(), _sample_win.height()));
            createFrameIntegralHOGC(&frame_hc, &frame_hog, sample_win);
//            qDebug() << "End Frame Integral Calculation: " << QTime::currentTime().toString("hh:mm:ss.zzz");

            //Build Object Features
            //Get the object rect relative to the sample window
            QPoint topLeft(_object.topLeft() - _sample_win.topLeft());
            QRect obj_rect_normalized = QRect(topLeft, _object.size());
            _obj_feature = getHOGCVal(&frame_hc, &frame_hog, obj_rect_normalized);


            //Calculate SBSR
            //Get the object point in reference to the sample win
            QPoint center_pt = _sample_win_top_left;  //If we add the _sample_win_top_left to (0,0) we get the mid point of the sample win which is the point we want.
            calculateSBSR(&_sbsr, &_samples, center_pt, &frame_hc, &frame_hog);
            _sample_count = _samples.size();


//            //Test that all weight is placed on the first sample
//            Sample sample;
//            sample.scale = scales[0];
//            sample.rect = _object;
//            sample.features = getHOGCVal(&frame_hc, &frame_hog, sample.rect);
//            _samples.append(sample);
//            Sample sample2;
//            sample2.scale = scales[0];
//            sample2.rect = QRect(start_x, start_y, obj_side, obj_side);
//            sample2.features = getHOGCVal(&frame_hc, &frame_hog, sample2.rect);
//            _samples.append(sample2);

            //TEST IF sbsr really makes the object
//            cv::Mat test = sample_set * sbsr;
//            std::cout << "child matrix:" << std::endl;
//            std::cout << sample_set * sbsr << std::endl;
//            std::cout << "original matrix:" << std::endl;
//            std::cout << _obj_feature << std::endl;

            //Setup the Kalman Filter
            int dynamic_vars = (_sample_count * 2) + 4; //2 phi arrays and 4 position values
            int measured_vars = _sample_count + 2;
            _k_filter = cv::KalmanFilter(dynamic_vars, measured_vars);
            cv::Mat transitions = cv::Mat::eye(dynamic_vars, dynamic_vars, CV_32F);
            int x=_sample_count+2;
            int y=0;
            while(x<(dynamic_vars)) {
                transitions.at<float>(x,y) = 1;
                x++;
                y++;
            }
            _k_filter.transitionMatrix = transitions;
            estimate = cv::Mat_<float>(dynamic_vars, 1);
            estimate_sbsr = cv::Mat_<float>(_sample_count, 1);
            estimate_pos = cv::Mat_<float>(2,1);
            prediction = cv::Mat_<float>(dynamic_vars, 1);
            prediction_sbsr = cv::Mat_<float>(_sample_count, 1);
            prediction_pos = cv::Mat_<float>(2,1);

            //TEST
            measurement = cv::Mat_<float>(measured_vars, 1);

            for(int i=0; i < dynamic_vars; i++) {
                if ( i < _sample_count ) {
                    double val = _sbsr.at<double>(i);
                    _k_filter.statePre.at<float>(i) = val;

                    //TEST
                    measurement.at<float>(i) = val;

                } else if ( i == _sample_count ) {
                    _k_filter.statePre.at<float>(i) = _object.center().x();
                    _k_filter.statePre.at<float>(i + 1) = _object.center().y();

                    //TEST
                    measurement.at<float>(i) = _object.center().x();
                    measurement.at<float>(i+1) = _object.center().y();

                    i++;
                } else {
                    _k_filter.statePre.at<float>(i) = 0;
                }
            }
            //Perform the first rounds of kalman filter
            _k_filter.correct(measurement);

//            std::cout << _k_filter.statePre << std::endl;

            cv::setIdentity(_k_filter.measurementMatrix);
            cv::setIdentity(_k_filter.processNoiseCov, cv::Scalar::all(1e-4));
            cv::setIdentity(_k_filter.measurementNoiseCov, cv::Scalar::all(1e-1));
            cv::setIdentity(_k_filter.errorCovPost, cv::Scalar::all(.1));

            _tracking = true;
        }

        _initialize = false;
        qDebug() << "End Initialized";

    }//end initialize

    //If tracking is requested then track

}

void AdaSR::initialize(QRect selection, cv::Mat img)
{
    _selection = selection;
    _original_img = img;
}

void AdaSR::track(cv::Mat img)
{

}


void AdaSR::createFrameIntegralHOGC(cv::Mat *hc_mat, cv::Mat *hog_mat, cv::Mat image)
{
//    qDebug() << "Start Integral HC: " << QTime::currentTime().toString("hh:mm:ss.zzz");

    *hc_mat = cv::Mat(image.rows, image.cols, custom::VecHC3::type);
    *hog_mat = cv::Mat(image.rows, image.cols, custom::VecHOG::type);

    //HC parameters
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, CV_BGR2GRAY);

    //HOG parameters
    cv::Mat grad_x, grad_y;
    int ddepth = CV_32F;
    cv::Sobel(gray_img, grad_x, ddepth, 1, 0, 3);
    cv::Sobel(gray_img, grad_y, ddepth, 0, 1, 3);
    float temp_gradient;

    cv::Vec3b pixel;
    //Loop through each pixel in the frame
    double angle_norm = 180 / M_PI;
    custom::VecHC3 pixel_hc;
    for( int y=0; y < image.rows; y++) {
        for( int x=0; x < image.cols; x++ ) {
            cv::Point point(x,y);

            pixel = image.at<cv::Vec3b>( point );
            int blue_bin = pixel[0] / 16;
            int green_bin = pixel[1] / 16;
            int red_bin = pixel[2] / 16 ;

            pixel_hc = custom::VecHC3::all(0);

            pixel_hc[red_bin]++;
            pixel_hc[green_bin + 16]++;
            pixel_hc[blue_bin + 32]++;

            //Sum up the hist values
            if ( x > 0 && y > 0 ) {
                pixel_hc += hc_mat->at<custom::VecHC3>(cv::Point(x-1, y)) +
                        hc_mat->at<custom::VecHC3>(cv::Point(x, y-1)) -
                        hc_mat->at<custom::VecHC3>(cv::Point(x-1, y-1));
            } else if ( x == 0 && y > 0 ) {
                pixel_hc += hc_mat->at<custom::VecHC3>(cv::Point(x, y-1));
            } else if ( x > 0 && y == 0 ) {
                pixel_hc += hc_mat->at<custom::VecHC3>(cv::Point(x-1, y));
            }

            //Save the hist values for this pixel
            hc_mat->at<custom::VecHC3>(point) = pixel_hc;

            //Perform HOG operation
            float dx = grad_x.at<float>(point);
            float dy = grad_y.at<float>(point);
            double grad_mag = qSqrt( qPow(dx,2) + qPow(dy,2) );
            double angle = qAtan2(dy, dx);

            custom::VecHOG pixel_hog = custom::VecHOG::all(0.0);
            if ( angle < 0 ) {
                temp_gradient = ((angle + M_PI + M_PI) * angle_norm);
                int grad_bin = qFloor(temp_gradient / 45);
                pixel_hog[grad_bin] = grad_mag;
            } else if( angle > 0 ) {
                temp_gradient = (angle * angle_norm);
                int grad_bin = qFloor(temp_gradient / 45);
                pixel_hog[grad_bin] = grad_mag;
            }

            //Add neighbor pixel gradients
            //Sum up the VecHogs
            if ( x > 0 && y > 0 ) {
                pixel_hog += hog_mat->at<custom::VecHOG>(cv::Point(x-1, y)) +
                        hog_mat->at<custom::VecHOG>(cv::Point(x, y-1)) -
                        hog_mat->at<custom::VecHOG>(cv::Point(x-1, y-1));
            } else if ( x == 0 && y > 0 ) {
                pixel_hog += hog_mat->at<custom::VecHOG>(cv::Point(x, y-1));
            } else if ( x > 0 && y == 0 ) {
                pixel_hog += hog_mat->at<custom::VecHOG>(cv::Point(x-1, y));
            }

            hog_mat->at<custom::VecHOG>(point) = pixel_hog;

        }

    }

//    qDebug() << "End Integral HC: " << QTime::currentTime().toString("hh:mm:ss.zzz");
}

custom::VecHC3 AdaSR::getHCVal(QRect rect, cv::Mat* hc_mat)
{
//    qDebug() << "Start get HC Val: " << QTime::currentTime().toString("hh:mm:ss.zzz");

    custom::VecHC3 A = hc_mat->at<custom::VecHC3>(cv::Point(rect.topLeft().x(), rect.topLeft().y()));
    custom::VecHC3 B = hc_mat->at<custom::VecHC3>(cv::Point(rect.topRight().x(), rect.topRight().y()));
    custom::VecHC3 C = hc_mat->at<custom::VecHC3>(cv::Point(rect.bottomRight().x(), rect.bottomRight().y()));
    custom::VecHC3 D = hc_mat->at<custom::VecHC3>(cv::Point(rect.bottomLeft().x(), rect.bottomLeft().y()));

    custom::VecHC3 sum = C + A - B - D;

//    qDebug() << "End get HC Val: " << QTime::currentTime().toString("hh:mm:ss.zzz");
    return sum;

}

custom::VecHOG9 AdaSR::getHOGVal(QRect rect, cv::Mat* hog_mat)
{
//    qDebug() << "Start get HOG Val: " << QTime::currentTime().toString("hh:mm:ss.zzz");
    //Every rect will be divisable by 4
    int half_side = qFloor(rect.width() / 2);
    int quarter_side = qFloor(half_side / 2);

    custom::VecHOG9 sum = custom::VecHOG9::all(0);

    //Loop through each block
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++) {
            //Get block points
            int left_x = (j * quarter_side) + rect.topLeft().x();
            int right_x = left_x + half_side-1;
            int top_y = (i * quarter_side) + rect.topLeft().y();
            int bot_y = top_y + half_side-1;

            custom::VecHOG A = hog_mat->at<custom::VecHOG>(cv::Point(left_x, top_y));
            custom::VecHOG B = hog_mat->at<custom::VecHOG>(cv::Point(right_x, top_y));
            custom::VecHOG C = hog_mat->at<custom::VecHOG>(cv::Point(right_x, bot_y));
            custom::VecHOG D = hog_mat->at<custom::VecHOG>(cv::Point(left_x, bot_y));

            custom::VecHOG temp = C + A - B - D;

            int index_offset = ((i * 3) + j) * 8; //Offset depending on the block number and bin size (8)
            for(int val=0; val < 8; val++) {
                sum[val+index_offset] = temp[val];
            }

        }
    }

//    qDebug() << "End get HOG Val: " << QTime::currentTime().toString("hh:mm:ss.zzz");
    return sum;
}

cv::Mat AdaSR::getHOGCVal(cv::Mat *hc_mat, cv::Mat *hog_mat, QRect rect)
{
    custom::VecHC3 hc = getHCVal(rect, hc_mat);
    custom::VecHOG9 hog = getHOGVal(rect, hog_mat);

//    custom::VecHOGC result = custom::VecHOGC::all(0);
    cv::Mat result = cv::Mat(120, 1, CV_64F, cv::Scalar(5));
    int i,j=0;
    for(i=0; i<48; i++) {
        result.at<double>(i,0) = hc[i];
    }
    for(j=0; j<72; j++) {
        result.at<double>(j+i,0) = hog[j];
    }
    return result;
}


void AdaSR::performMinimization(cv::Mat *result, QList<Sample> *samples, cv::Mat *obj_feature)
{

    try {
        GRBModel model(env);

        //Initialize variable arrays
        double b = 0.1;
        const int quad_size = samples->size() * 3;
        const int lin_size = _samples.size() * 2;
        GRBVar* linear_indexes = new GRBVar[lin_size];
        double* linear_values = new double[lin_size];
        GRBVar* quad_row_index = new GRBVar[quad_size];
        GRBVar* quad_col_index = new GRBVar[quad_size];
        double* quad_values = new double[quad_size];

        //Calculate the right hand side
        cv::Mat obj_feature_sqrd;
        cv::mulTransposed(*obj_feature, obj_feature_sqrd, true);
//        std::cout << "Obj Feat: " << _obj_feature << std::endl;
//        std::cout << "Obj Feat Sqrd: " << obj_feature_sqrd << std::endl;
//        std::cout << "Val: " << obj_feature_sqrd.at<double>(0,0) << std::endl;
//        b += obj_feature_sqrd.at<double>(cv::Point(0,0));

        //Matrix multiplication vars
        cv::Mat sample_feature_sqrd;
        cv::Mat sample_feature_trans;

        //Objective Minimization Vars
        double* coef = new double[lin_size];
        std::fill_n(coef, lin_size, 1);
        GRBVar* vars = new GRBVar[lin_size];

        int lin_index = 0;
        int quad_index = 0;
        int index = 0;
        foreach(Sample sample, *samples) {
            cv::Mat feature = sample.features;

            //Initialize the variable indexes
            int x_pos = index;
            int x_neg = index+1;
            GRBVar xpos = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_SEMICONT, "x+"+ QString::number(x_pos).toStdString());
            GRBVar xneg = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_SEMICONT, "x-"+ QString::number(x_neg).toStdString());
//            vars[x_pos]
//            vars[x_neg] =

            // Ai^2Xi^2 - 2AiXiF + Fi^2  <= 0.1^2
            // This can be rewritten as
            // Ait*AiXi^2 - 2FtAiXi <= 0.1^2 - Ft*F
            cv::transpose(feature, sample_feature_trans); // A transpose
            cv::Mat temp_mat = sample_feature_trans * *obj_feature;// A transpose * F
            cv::mulTransposed(feature, sample_feature_sqrd, true); // A transpose * A

            double at_f = temp_mat.at<double>(0,0);
            double at_a = sample_feature_sqrd.at<double>(0,0);; // A transpose * A

//            std::cout << "Obj Feat: " << temp_mat << std::endl;
//            std::cout << "Obj Feat Sqrd: " << sample_feature_sqrd << std::endl;
//            std::cout << "Val: " << sample_feature_sqrd.at<double>(0,0) << std::endl;
//            std::cout << "X: " << at_f << std::endl;
//            std::cout << "J: " << at_a << std::endl;
//            return;

            //Set the linear value
            //-2AtFx+ + 2AtFx-
            linear_indexes[lin_index] = xpos;
            linear_values[lin_index] = -2 * at_f;
            lin_index++;
            linear_indexes[lin_index] = xneg;
            linear_values[lin_index] = 2 * at_f;
            lin_index++;


            //Set the Quadratic value
            //AtAx+^2 - 2AtAx+x- + AtAx-^2
            quad_row_index[quad_index] = xpos;
            quad_col_index[quad_index] = xpos;
            quad_values[quad_index] = at_a;
            quad_index++;
            quad_row_index[quad_index] = xpos;
            quad_col_index[quad_index] = xneg;
            quad_values[quad_index] = -2 * at_a;
            quad_index++;
            quad_row_index[quad_index] = xneg;
            quad_col_index[quad_index] = xneg;
            quad_values[quad_index] = at_a;
            quad_index++;


            //Move to next sample
            index += 2;
        }

        model.update();

        GRBLinExpr obj;
        obj.addTerms(coef, linear_indexes, lin_size);
        model.setObjective(obj, GRB_MINIMIZE); //Minimize the variables

        GRBQuadExpr qexpr;
        qexpr.addTerms(linear_values, linear_indexes, lin_size);
        qexpr.addTerms(quad_values, quad_row_index, quad_col_index, quad_size);
        qexpr.addConstant( obj_feature_sqrd.at<double>(0,0) );

        model.addQConstr(qexpr, GRB_LESS_EQUAL, b, "main");

        model.optimize();

        int k=0;
        *result = cv::Mat(samples->size(), 1, CV_64F);
        for(int i=0; i < lin_size; i+=2) {
            double test = linear_indexes[i].get(GRB_DoubleAttr_X) - linear_indexes[i+1].get(GRB_DoubleAttr_X);
            if ( test < 0.00001 ) {
                test = 0;
            }
            result->at<double>(k, 0) = test;
            if ( test >= 0.05 )
                std::cout << "X" << k << " = " << test << std::endl;
            k++;
        }

//        std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

        //Clean up the dynamically created arrays
        delete[] linear_indexes;
        delete[] linear_values;
        delete[] quad_row_index;
        delete[] quad_col_index;
        delete[] quad_values;
        delete[] coef;
        delete[] vars;


    } catch(GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    } catch(...) {
        std::cout << "Exception during optimization" << std::endl;
    }
}

/**
  * point is relative to the respective image
  *
  */
void AdaSR::calculateSBSR(cv::Mat *sbsr_dst, QList<Sample> *sample_dst, QPoint point, cv::Mat *hc_mat, cv::Mat *hog_mat)
{
    //Assume that the img is large enough to contain the sample window

    //Using the object size create the sample window around this point
    QPoint top_left = point - _sample_win_top_left;
    QRect win_rect(top_left, QSize(_sample_win_side, _sample_win_side));

    //Create the samples
//    qDebug() << "Start Samples: " << QTime::currentTime().toString("hh:mm:ss.zzz");
    cv::Mat sample_set(120, 100, CV_64F);
    double scales[5] = {1.0,0.9,1.1,0.8,1.2}; // the first 3 will be chosen more often since this isn't uniform
    int col = 0;
    for(int y=0; y < win_rect.height() - _object.width(); y+=_d ) {
        for ( int x=0; x < win_rect.width() - _object.width(); x+=_d ) {
            int scale_index = 0;//0 + (rand() % (int)(4 - 0 + 1));
            int side = qFloor(_object.width() * scales[scale_index]);
            QRect sample_rect = QRect(x, y, side, side).intersect(win_rect);
            Sample sample;
            sample.scale = scales[scale_index];
            sample.rect = sample_rect;
            sample.features = getHOGCVal(hc_mat, hog_mat, sample.rect);
            sample_dst->append(sample);
            for ( int i=0; i < sample.features.rows; i++ ) {
                sample_set.at<double>(i, col) = sample.features.at<double>(i,0);
            }
            col++;
        }
    }

    //Get SBSR
    performMinimization(sbsr_dst, sample_dst, &_obj_feature);
}

double AdaSR::calcL1Norm(cv::Mat *mat1, cv::Mat *mat2)
{
    cv::Mat diff_mat = *mat1 - *mat2;
    double res = cv::norm(diff_mat, 2);
    return res;
}

bool AdaSR::constructSampleWindow()
{
    //Create a square object rectangle that has the same center point as the selection
    QPoint center = _original_selection.center();
    int side_len = qMin(_original_selection.width(), _original_selection.height());
    int obj_side = side_len - (side_len % 5);
    int obj_half_side = obj_side/2;
    //QT uses zero indexing but integral image does not we will have to modify the
    //integral image algorithm to handle this
    QPoint obj_topleft = center - QPoint(obj_half_side, obj_half_side);
    QRect obj_rect(obj_topleft, QSize(obj_side, obj_side));

    //Create the sample window
    _d = qMax(qFloor(obj_rect.width() / 5), 2);
    _sample_win_top_left = QPoint((_d * 5) + obj_half_side, (_d * 5) + obj_half_side);
    _sample_win_side = _d * 10 + obj_side;
    QPoint win_topleft = center - _sample_win_top_left;
    QRect win_rect(win_topleft, QSize(_sample_win_side, _sample_win_side));


    //Make sure the object and sample window are within the frame
    if ( _view_rect.contains(obj_rect) && _view_rect.contains(win_rect) ) {

        //Save rect
        _object = obj_rect;
        _sample_win = win_rect;

        return true;
    } else {
        return false;
    }
}
