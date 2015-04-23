uniform sampler2D image;

vec4 blur(vec2 step, vec2 center)
{

   vec4 sum = vec4(0.0);
 
   // blur in x (horizontal)
   // take three samples, with the distance blurSize between them

   vec4 binaryImage[3];

   binaryImage[0] = texture2D(image, vec2(center.s - step.s, center.t));
   binaryImage[1] = texture2D(image, vec2(center.s, center.t));
   binaryImage[2] = texture2D(image, vec2(center.s + step.s, center.t));

   for(int n = 0; n < 3; n++) {
       float grayLevel = (binaryImage[n].r + binaryImage[n].g + binaryImage[n].b)/3;
	   if(grayLevel > 0.1)
	       binaryImage[n] = vec4(1, 1, 1, 0);
	   else
	       binaryImage[n] = vec4(0, 0, 0, 0);
   }

   sum += binaryImage[0] * 0.25;
   sum += binaryImage[1] * 0.5;
   sum += binaryImage[2] * 0.25;

   return sum;

 }

void main (void)  
{

	vec2 step, center;
	step.s = 1.0/640.0;
	step.t = 1.0/480.0;
	center.s = gl_TexCoord[0].s;
	center.t = 1.0 - gl_TexCoord[0].t;
	gl_FragColor = blur(step, center);

}