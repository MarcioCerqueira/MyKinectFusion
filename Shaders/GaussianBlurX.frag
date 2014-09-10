uniform sampler2D image;

vec4 blur(vec2 step, vec2 center)
{

   vec4 sum = vec4(0.0);
 
   // blur in y (vertical)
   // take three samples, with the distance blurSize between them

   sum += texture2D(image, vec2(center.s - step.s, center.t)) * 0.25;
   sum += texture2D(image, vec2(center.s, center.t)) * 0.5;
   sum += texture2D(image, vec2(center.s + step.s, center.t)) * 0.25;

   return sum;

 }

void main (void)  
{

	vec2 step, center;
	step.s = 1.0/640.0;
	step.t = 1.0/480.0;
	center.s = gl_TexCoord[0].s;
	center.t = gl_TexCoord[0].t;
	gl_FragColor = blur(step, center);

}