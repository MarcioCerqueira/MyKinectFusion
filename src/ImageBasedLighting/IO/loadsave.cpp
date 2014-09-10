/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//
//  Loading and saving IPL images.
//

#include "ImageBasedLighting/IO/loadsave.h"

#undef min
#undef max

#include <iostream>

/****************************************************************************************\
*                                      Image Codecs                                      *
\****************************************************************************************/
namespace cv
{

enum { LOAD_CVMAT=0, LOAD_IMAGE=1, LOAD_MAT=2 };

static void*
hdrImread_( const String& filename, int flags, int hdrtype, Mat* mat=0 )
{
    IplImage* image = 0;
    CvMat *matrix = 0;
    Mat temp, *data = &temp;

    HdrDecoder *decoder = new HdrDecoder();
    if( !decoder )
        return 0;
    decoder->setSource(filename);
    if( !decoder->readHeader() )
        return 0;
    CvSize size;
    size.width = decoder->width();
    size.height = decoder->height();

    int type = decoder->type();
    if( flags != -1 )
    {
        if( (flags & CV_LOAD_IMAGE_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & CV_LOAD_IMAGE_COLOR) != 0 ||
           ((flags & CV_LOAD_IMAGE_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }

    if( hdrtype == LOAD_CVMAT || hdrtype == LOAD_MAT )
    {
        if( hdrtype == LOAD_CVMAT )
        {
            matrix = cvCreateMat( size.height, size.width, type );
            temp = cvarrToMat(matrix);
        }
        else
        {
            mat->create( size.height, size.width, type );
            data = mat;
        }
    }
    else
    {
        image = cvCreateImage( size, cvIplDepth(type), CV_MAT_CN(type) );
        temp = cvarrToMat(image);
    }

    if( !decoder->readData( *data ))
    {
        cvReleaseImage( &image );
        cvReleaseMat( &matrix );
        if( mat )
            mat->release();
        return 0;
    }

    return hdrtype == LOAD_CVMAT ? (void*)matrix :
        hdrtype == LOAD_IMAGE ? (void*)image : (void*)mat;
}

Mat hdrImread( const String& filename, int flags )
{
    Mat img;
    hdrImread_( filename, flags, LOAD_MAT, &img );
    return img;
}

static bool hdrImwrite_( const String& filename, const Mat& image,
                      const std::vector<int>& params, bool flipv )
{
    Mat temp;
    const Mat* pimage = &image;

    CV_Assert( image.channels() == 1 || image.channels() == 3 || image.channels() == 4 );

    HdrEncoder *encoder = new HdrEncoder();
    if( !encoder )
        CV_Error( CV_StsError, "could not find a writer for the specified extension" );
    if( !encoder->isFormatSupported(image.depth()) )
    {
        CV_Assert( encoder->isFormatSupported(CV_8U) );
        image.convertTo( temp, CV_8U );
        pimage = &temp;
    }

    if( flipv )
    {
        flip(*pimage, temp, 0);
        pimage = &temp;
    }

    encoder->setDestination( filename );
    bool code = encoder->write( *pimage, params );

    //    CV_Assert( code );
    return code;
}

bool hdrImwrite( const String& filename, InputArray _img,
              const std::vector<int>& params )
{
    Mat img = _img.getMat();
    return hdrImwrite_(filename, img, params, false);
}

static void*
hdrImdecode_( const Mat& buf, int flags, int hdrtype, Mat* mat=0 )
{
    CV_Assert(buf.data && buf.isContinuous());
    IplImage* image = 0;
    CvMat *matrix = 0;
    Mat temp, *data = &temp;
    String filename;

    HdrDecoder *decoder = new HdrDecoder();
    if( !decoder )
        return 0;

    if( !decoder->setSource(buf) )
    {
        filename = tempfile();
        FILE* f = fopen( filename.c_str(), "wb" );
        if( !f )
            return 0;
        size_t bufSize = buf.cols*buf.rows*buf.elemSize();
        fwrite( &buf.data[0], 1, bufSize, f );
        fclose(f);
        decoder->setSource(filename);
    }

    if( !decoder->readHeader() )
    {
        if( !filename.empty() )
            remove(filename.c_str());
        return 0;
    }

    CvSize size;
    size.width = decoder->width();
    size.height = decoder->height();

    int type = decoder->type();
    if( flags != -1 )
    {
        if( (flags & CV_LOAD_IMAGE_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & CV_LOAD_IMAGE_COLOR) != 0 ||
           ((flags & CV_LOAD_IMAGE_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }

    if( hdrtype == LOAD_CVMAT || hdrtype == LOAD_MAT )
    {
        if( hdrtype == LOAD_CVMAT )
        {
            matrix = cvCreateMat( size.height, size.width, type );
            temp = cvarrToMat(matrix);
        }
        else
        {
            mat->create( size.height, size.width, type );
            data = mat;
        }
    }
    else
    {
        image = cvCreateImage( size, cvIplDepth(type), CV_MAT_CN(type) );
        temp = cvarrToMat(image);
    }

    bool code = decoder->readData( *data );
    if( !filename.empty() )
        remove(filename.c_str());

    if( !code )
    {
        cvReleaseImage( &image );
        cvReleaseMat( &matrix );
        if( mat )
            mat->release();
        return 0;
    }

    return hdrtype == LOAD_CVMAT ? (void*)matrix :
        hdrtype == LOAD_IMAGE ? (void*)image : (void*)mat;
}


Mat hdrImdecode( InputArray _buf, int flags )
{
    Mat buf = _buf.getMat(), img;
    hdrImdecode_( buf, flags, LOAD_MAT, &img );
    return img;
}

Mat hdrImdecode( InputArray _buf, int flags, Mat* dst )
{
    Mat buf = _buf.getMat(), img;
    dst = dst ? dst : &img;
    hdrImdecode_( buf, flags, LOAD_MAT, dst );
    return *dst;
}

bool hdrImencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf, const std::vector<int>& params )
{
    Mat image = _image.getMat();

    int channels = image.channels();
    CV_Assert( channels == 1 || channels == 3 || channels == 4 );

    HdrEncoder *encoder = new HdrEncoder();
    if( !encoder )
        CV_Error( CV_StsError, "could not find encoder for the specified extension" );

    if( !encoder->isFormatSupported(image.depth()) )
    {
        CV_Assert( encoder->isFormatSupported(CV_8U) );
        Mat temp;
        image.convertTo(temp, CV_8U);
        image = temp;
    }

    bool code;
    if( encoder->setDestination(buf) )
    {
        code = encoder->write(image, params);
        encoder->throwOnEror();
        CV_Assert( code );
    }
    else
    {
        String filename = tempfile();
        code = encoder->setDestination(filename);
        CV_Assert( code );

        code = encoder->write(image, params);
        encoder->throwOnEror();
        CV_Assert( code );

        FILE* f = fopen( filename.c_str(), "rb" );
        CV_Assert(f != 0);
        fseek( f, 0, SEEK_END );
        long pos = ftell(f);
        buf.resize((size_t)pos);
        fseek( f, 0, SEEK_SET );
        buf.resize(fread( &buf[0], 1, buf.size(), f ));
        fclose(f);
        remove(filename.c_str());
    }
    return code;
}

IplImage* cvLoadHDRImage( const char* filename, int iscolor )
{
    return (IplImage*)cv::hdrImread_(filename, iscolor, cv::LOAD_IMAGE );
}


CvMat* cvLoadHDRImageM( const char* filename, int iscolor )
{
    return (CvMat*)cv::hdrImread_( filename, iscolor, cv::LOAD_CVMAT );
}

int cvSaveHDRImage( const char* filename, const CvArr* arr, const int* _params )
{
    int i = 0;
    if( _params )
    {
        for( ; _params[i] > 0; i += 2 )
            ;
    }
    return cv::hdrImwrite_(filename, cv::cvarrToMat(arr),
        i > 0 ? std::vector<int>(_params, _params+i) : std::vector<int>(),
        CV_IS_IMAGE(arr) && ((const IplImage*)arr)->origin == IPL_ORIGIN_BL );
}

/*
CV_IMPL IplImage*
cvDecodeImage( const CvMat* _buf, int iscolor )
{
    CV_Assert( _buf && CV_IS_MAT_CONT(_buf->type) );
    cv::Mat buf(1, _buf->rows*_buf->cols*CV_ELEM_SIZE(_buf->type), CV_8U, _buf->data.ptr);
    return (IplImage*)cv::imdecode_(buf, iscolor, cv::LOAD_IMAGE );
}

CV_IMPL CvMat*
cvDecodeImageM( const CvMat* _buf, int iscolor )
{
    CV_Assert( _buf && CV_IS_MAT_CONT(_buf->type) );
    cv::Mat buf(1, _buf->rows*_buf->cols*CV_ELEM_SIZE(_buf->type), CV_8U, _buf->data.ptr);
    return (CvMat*)cv::imdecode_(buf, iscolor, cv::LOAD_CVMAT );
}

CV_IMPL CvMat*
cvEncodeImage( const char* ext, const CvArr* arr, const int* _params )
{
    int i = 0;
    if( _params )
    {
        for( ; _params[i] > 0; i += 2 )
            ;
    }
    cv::Mat img = cv::cvarrToMat(arr);
    if( CV_IS_IMAGE(arr) && ((const IplImage*)arr)->origin == IPL_ORIGIN_BL )
    {
        cv::Mat temp;
        cv::flip(img, temp, 0);
        img = temp;
    }
    std::vector<uchar> buf;

    bool code = cv::imencode(ext, img, buf,
        i > 0 ? std::vector<int>(_params, _params+i) : std::vector<int>() );
    if( !code )
        return 0;
    CvMat* _buf = cvCreateMat(1, (int)buf.size(), CV_8U);
    memcpy( _buf->data.ptr, &buf[0], buf.size() );

    return _buf;
}
*/
}

/****************************************************************************************\
*                         HighGUI loading & saving function implementation               *
\****************************************************************************************/

/*
int cvHaveImageReader( const char* filename )
{
	cv::HdrDecoder decoder;
    return !decoder.empty();
}

CV_IMPL int cvHaveImageWriter( const char* filename )
{
    cv::HdrEncoder encoder = cv::findEncoder(filename);
    return !encoder.empty();
}

IplImage* cvLoadImage( const char* filename, int iscolor )
{
    return (IplImage*)cv::imread_(filename, iscolor, cv::LOAD_IMAGE );
}

CvMat* cvLoadImageM( const char* filename, int iscolor )
{
    return (CvMat*)cv::imread_( filename, iscolor, cv::LOAD_CVMAT );
}

int cvSaveImage( const char* filename, const CvArr* arr, const int* _params )
{
    int i = 0;
    if( _params )
    {
        for( ; _params[i] > 0; i += 2 )
            ;
    }
    return cv::imwrite_(filename, cv::cvarrToMat(arr),
        i > 0 ? std::vector<int>(_params, _params+i) : std::vector<int>(),
        CV_IS_IMAGE(arr) && ((const IplImage*)arr)->origin == IPL_ORIGIN_BL );
}

CV_IMPL IplImage*
cvDecodeImage( const CvMat* _buf, int iscolor )
{
    CV_Assert( _buf && CV_IS_MAT_CONT(_buf->type) );
    cv::Mat buf(1, _buf->rows*_buf->cols*CV_ELEM_SIZE(_buf->type), CV_8U, _buf->data.ptr);
    return (IplImage*)cv::imdecode_(buf, iscolor, cv::LOAD_IMAGE );
}

CV_IMPL CvMat*
cvDecodeImageM( const CvMat* _buf, int iscolor )
{
    CV_Assert( _buf && CV_IS_MAT_CONT(_buf->type) );
    cv::Mat buf(1, _buf->rows*_buf->cols*CV_ELEM_SIZE(_buf->type), CV_8U, _buf->data.ptr);
    return (CvMat*)cv::imdecode_(buf, iscolor, cv::LOAD_CVMAT );
}

CV_IMPL CvMat*
cvEncodeImage( const char* ext, const CvArr* arr, const int* _params )
{
    int i = 0;
    if( _params )
    {
        for( ; _params[i] > 0; i += 2 )
            ;
    }
    cv::Mat img = cv::cvarrToMat(arr);
    if( CV_IS_IMAGE(arr) && ((const IplImage*)arr)->origin == IPL_ORIGIN_BL )
    {
        cv::Mat temp;
        cv::flip(img, temp, 0);
        img = temp;
    }
    std::vector<uchar> buf;

    bool code = cv::imencode(ext, img, buf,
        i > 0 ? std::vector<int>(_params, _params+i) : std::vector<int>() );
    if( !code )
        return 0;
    CvMat* _buf = cvCreateMat(1, (int)buf.size(), CV_8U);
    memcpy( _buf->data.ptr, &buf[0], buf.size() );

    return _buf;
}
*/
/* End of file. */
