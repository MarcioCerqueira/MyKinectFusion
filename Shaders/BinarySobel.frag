uniform sampler2D image;

float intensity(vec4 color)
{
	return sqrt((color.x*color.x)+(color.y*color.y)+(color.z*color.z));
}

vec4 sobel(vec2 step, vec2 center)
{

	//binarize image
	vec4 binaryImage[8];
	binaryImage[0] = texture2D(image,center + vec2(-step.s,step.t));
    binaryImage[1] = texture2D(image,center + vec2(-step.s,0));
    binaryImage[2] = texture2D(image,center + vec2(-step.s,-step.t));
    binaryImage[3] = texture2D(image,center + vec2(0,step.t));
    binaryImage[4] = texture2D(image,center + vec2(0,-step.t));
    binaryImage[5] = texture2D(image,center + vec2(step.s,step.t));
    binaryImage[6] = texture2D(image,center + vec2(step.s,0));
    binaryImage[7] = texture2D(image,center + vec2(step.s,-step.t));

	for(int n = 0; n < 8; n++) {
		float grayLevel = (binaryImage[n].r + binaryImage[n].g + binaryImage[n].b)/3;
		if(grayLevel > 0.1)
				binaryImage[n] = vec4(1, 1, 1, 1);
			else
				binaryImage[n] = vec4(0, 0, 0, 0);
	}

	// get samples around pixel
    float tleft = intensity(binaryImage[0]);
    float left = intensity(binaryImage[1]);
    float bleft = intensity(binaryImage[2]);
    float top = intensity(binaryImage[3]);
    float bottom = intensity(binaryImage[4]);
    float tright = intensity(binaryImage[5]);
    float right = intensity(binaryImage[6]);
    float bright = intensity(binaryImage[7]);

	// Sobel masks (to estimate gradient)
	//        1 0 -1     -1 -2 -1
	//    X = 2 0 -2  Y = 0  0  0
	//        1 0 -1      1  2  1

    float x = tleft + 2.0*left + bleft - tright - 2.0*right - bright;
    float y = -tleft - 2.0*top - tright + bleft + 2.0 * bottom + bright;
    float color = sqrt((x*x) + (y*y));
    if (color >= 1.0) 
		return vec4(1.0,1.0,1.0,0.0);
    return vec4(0.0,0.0,0.0,0.0);

 }

void main (void)  
{

	vec2 step, center;
	step.s = 1.0/960.0;
	step.t = 1.0/720.0;
	center.s = gl_TexCoord[0].s;
	center.t = gl_TexCoord[0].t;
	gl_FragColor = sobel(step, center);

}