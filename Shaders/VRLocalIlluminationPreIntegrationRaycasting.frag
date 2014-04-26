uniform sampler3D volume;
uniform sampler3D minMaxOctree;
uniform sampler2D transferFunction;
uniform sampler2D noise;
uniform sampler2D frontFrameBuffer;
uniform sampler2D backFrameBuffer;
uniform float stepSize;
uniform int clippingPlane;
uniform int inverseClipping;
uniform int clippingOcclusion;
uniform float clippingPlaneLeftX;
uniform float clippingPlaneRightX;
uniform float clippingPlaneUpY;
uniform float clippingPlaneDownY;
uniform float clippingPlaneFrontZ;
uniform float clippingPlaneBackZ;
uniform float earlyRayTerminationThreshold;
uniform vec3 camera;
varying vec3 v;
uniform int MIP;
uniform int forwardDifference;
uniform int stochasticJithering;
uniform int windowWidth;
uniform int windowHeight;

//Blinn-Phong Illumination
vec3 BlinnPhongShading(vec3 L, vec3 N, vec3 V) 
{

	vec3 H = normalize(L + V);
	
	vec4 ambient = gl_FrontLightProduct[0].ambient;
	
	float diffuseLight = max(dot(L, N), 0.0);
	vec4 diffuse = gl_FrontLightProduct[0].diffuse * diffuseLight;

	float specularLight = pow(max(dot(H, N), 0.0), 60);
	if(diffuseLight <= 0.0) specularLight = 0.0;
	vec4 specular = gl_FrontLightProduct[0].specular * specularLight;

	return vec3(gl_FrontLightModelProduct.sceneColor + ambient + diffuse + specular);
}

vec3 PhongShading(vec3 L, vec3 N, vec3 V) 
{
	
   
   vec3 E = normalize(-V); // we are in Eye Coordinates, so EyePos is (0,0,0)  
   vec3 R = normalize(-reflect(L,N));  
 
   //calculate Ambient Term:  
   vec4 Iamb = gl_FrontLightProduct[0].ambient;    

   //calculate Diffuse Term:  
   vec4 Idiff = gl_FrontLightProduct[0].diffuse * max(dot(N,L), 0.0);    
   
   // calculate Specular Term:
   vec4 Ispec = gl_FrontLightProduct[0].specular 
                * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);

   // write Total Color:  
   return vec3(gl_FrontLightModelProduct.sceneColor + Iamb + Idiff + Ispec); 
   
}

vec4 computeIllumination(vec4 scalar, vec3 position) 
{

	if(scalar.a > 0.075) {
		float delta = 0.01;
		vec3 sample1, sample2;
		vec3 alpha1 = vec3(0, 0, 0);
		vec3 alpha2 = vec3(0, 0, 0);
		sample2.x = texture3D(volume, vec3(position + vec3(delta, 0, 0))).x;
		sample2.y = texture3D(volume, vec3(position + vec3(0, delta, 0))).x;
		sample2.z = texture3D(volume, vec3(position + vec3(0, 0, delta))).x;
		alpha2.x = texture2D(transferFunction, vec2(sample2.x, sample2.x)).a;
		alpha2.y = texture2D(transferFunction, vec2(sample2.y, sample2.y)).a;
		alpha2.z = texture2D(transferFunction, vec2(sample2.z, sample2.z)).a;
		
		if(forwardDifference == 0) {
			sample1.x = texture3D(volume, vec3(position - vec3(delta, 0, 0))).x;
			sample1.y = texture3D(volume, vec3(position - vec3(0, delta, 0))).x;
			sample1.z = texture3D(volume, vec3(position - vec3(0, 0, delta))).x;
			alpha1.x = texture2D(transferFunction, vec2(sample1.x, sample1.x)).a;
			alpha1.y = texture2D(transferFunction, vec2(sample1.y, sample1.y)).a;
			alpha1.z = texture2D(transferFunction, vec2(sample1.z, sample1.z)).a;
		}

		//central difference and normalization
		//for(int i = 0; i < 1; i++) {
			vec3 N;
			if(forwardDifference == 1)
				N = normalize(vec3(scalar) - alpha2);
			else
				N = normalize(alpha2 - alpha1);
			vec3 L = normalize(gl_LightSource[0].position.xyz - v); 
			vec3 V = normalize(-v);
			scalar.rgb += BlinnPhongShading(L, N, V).rgb;
		//}
	}

	return scalar;
}

bool checkClippingPlane(vec3 position) 
{
	if(position.x >= clippingPlaneLeftX && position.x < clippingPlaneRightX 
	&& position.y >= clippingPlaneDownY && position.y < clippingPlaneUpY
	&& position.z >= clippingPlaneFrontZ && position.z < clippingPlaneBackZ)
		return false;
	else
		return true;
}

void main (void)  
{

	vec4 value;
	vec2 scalar = vec2(0, 0);
	vec4 src = vec4(0, 0, 0, 0);
	vec4 rayStart = texture2D(frontFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	vec4 rayEnd = texture2D(backFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	if(rayStart == rayEnd)
		discard;
	//Initialize accumulated color and opacity
	vec4 dst = vec4(0, 0, 0, 0);
	//Determine volume entry position
	vec3 position = vec3(rayStart);
	vec3 direction = vec3(rayEnd) - vec3(rayStart);
	float len = length(direction); // the length from front to back is calculated and used to terminate the ray
    direction = normalize(direction);
	if(stochasticJithering == 1)
		position = position + direction * texture2D(noise, gl_FragCoord.xy / 256.0).x/64.0;
	float dirLength = length(direction);
	//Loop for ray traversal
	float maxStepSize = 0.04;
	//float maxStepSize = 4 * stepSize;
	float accLength = 0.0;
	vec4 maxOpacity;
	bool clip = false;
	bool firstHit = false;

	for(int i = 0; i < 200; i++) //Some large number
	{
		
		maxOpacity = texture3D(minMaxOctree, position);
		
		if(clippingPlane) { 
		
			clip = checkClippingPlane(position);
			if(inverseClipping) clip = !clip;
			
			if(clippingOcclusion) {

				if(!firstHit) {
					value = texture3D(volume, position);
					if(value.a > 0.1) {
						if(clip) return;
						else firstHit = true;
					}
				}

			}

		}

		if(maxOpacity.g > 0.0) {
			
			if(!clip) {
		
				//Data access to scalar value in 3D volume texture
				value = texture3D(volume, position);

				scalar.y = value.a;
				src = texture2D(transferFunction, scalar.xy);
				src = computeIllumination(src, position);
			
				//Front-to-back compositing
				if(MIP == 0) {
					if(src.a > 0.1)
						dst = (1.0 - dst.a) * src + dst;
				} else
					dst = max(dst, src);
	
				//Save previous scalar value
				scalar.x = src.a;

			}
				
			//Advance ray position along ray direction
			if(clippingOcclusion) {
			
				if(firstHit) {
					position = position + direction * stepSize;
					accLength += dirLength * stepSize;
				} else {
					position = position + direction * 0.008;
					accLength += dirLength * 0.008;
				}
			
			} else {
			
				position = position + direction * stepSize;
				accLength += dirLength * stepSize;
			
			}
			
		} else {
		
			position = position + direction * maxStepSize;
			accLength += dirLength * maxStepSize;
		
		}
		
		//Additional termination condition for early ray termination
		if(dst.a > earlyRayTerminationThreshold)
			break;
		
		//Ray termination: Test if outside volume...
		if(accLength > len)
			break;
		
	}

	gl_FragColor = dst;
}