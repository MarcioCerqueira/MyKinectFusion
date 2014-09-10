uniform sampler3D volume;
uniform sampler3D minMaxOctree;
uniform sampler2D transferFunction;
uniform sampler2D noise;
uniform sampler2D frontFrameBuffer;
uniform sampler2D backFrameBuffer;

uniform float stepSize;
uniform float earlyRayTerminationThreshold;
uniform float isosurfaceThreshold;

uniform int clippingPlane;
uniform int inverseClipping;
uniform int clippingOcclusion;
uniform float clippingPlaneLeftX;
uniform float clippingPlaneRightX;
uniform float clippingPlaneUpY;
uniform float clippingPlaneDownY;
uniform float clippingPlaneFrontZ;
uniform float clippingPlaneBackZ;

uniform int transferFunctionOn;
uniform int BlinnPhongShadingOn;
uniform int NonPolygonalIsoSurfaceOn;
uniform int stochasticJithering;
uniform int triCubicInterpolation;
uniform int MIP;

varying vec3 v;
uniform int forwardDifference;
uniform int useIBL;
uniform float kt;
uniform float ks;

uniform int windowWidth;
uniform int windowHeight;

//Spherical Harmonics values
uniform vec3 L00;
uniform vec3 L1m1;
uniform vec3 L10;
uniform vec3 L11;
uniform vec3 L2m2;
uniform vec3 L2m1;
uniform vec3 L20;
uniform vec3 L21;
uniform vec3 L22;
uniform vec3 lightDir;
uniform vec3 lightColor;
const float C1 = 0.429043;
const float C2 = 0.511664;
const float C3 = 0.743125;
const float C4 = 0.886227;
const float C5 = 0.247708;
uniform float diffuseScaleFactor;
uniform float specularScaleFactor;
uniform float shininess;

vec3 diffuseIBL(vec3 N)
{

   vec3 diffuseColor = C1 * L22 * (N.x * N.x - N.y * N.y) +
		C3 * L20 * N.z * N.z +
		C4 * L00 -
		C5 * L20 +
		2.0 * C1 * L2m2 * N.x * N.y +
		2.0 * C1 * L21 * N.x * N.z +
		2.0 * C1 * L2m1 * N.y * N.z +
		2.0 * C2 * L11 * N.x +
		2.0 * C2 * L1m1 * N.y +
		2.0 * C2 * L10 * N.z;

   //calculate Diffuse Term:  
   vec4 Idiff = diffuseScaleFactor * vec4(diffuseColor, 1.0);

   // write Total Color:  
   return vec3(Idiff);  

}

vec3 specularIBL(vec3 N, vec3 V)
{

   vec3 E = normalize(-V); // we are in Eye Coordinates, so EyePos is (0,0,0)  
   vec3 R = normalize(-reflect(vec3(lightDir.x, -lightDir.y, lightDir.z), N));  
   vec4 Ispec = vec4(lightColor, 1.0) * pow(max(dot(R,V),0.0), 0.3 * shininess) * specularScaleFactor;
   return vec3(Ispec);

}

//Blinn-Phong Illumination
vec3 BlinnPhongShading(vec3 L, vec3 N, vec3 V) 
{

	vec3 H = normalize(L + V);
	
	vec4 ambient = gl_FrontLightProduct[0].ambient;
	
	float diffuseLight = max(dot(L, N), 0.0);
	vec4 diffuse = (gl_FrontLightProduct[0].diffuse) * diffuseLight;

	float specularLight = pow(max(dot(H, N), 0.0), 60);
	if(diffuseLight <= 0.0) specularLight = 0.0;
	vec4 specular = gl_FrontLightProduct[0].specular * specularLight;

	return vec3(gl_FrontLightModelProduct.sceneColor + ambient + diffuse + specular);
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

		if(transferFunctionOn == 1) {
			
			alpha2.x = texture2D(transferFunction, vec2(sample2.x, sample2.x)).a;
			alpha2.y = texture2D(transferFunction, vec2(sample2.y, sample2.y)).a;
			alpha2.z = texture2D(transferFunction, vec2(sample2.z, sample2.z)).a;
			
		}

		if(forwardDifference == 0) {
			
			sample1.x = texture3D(volume, vec3(position - vec3(delta, 0, 0))).x;
			sample1.y = texture3D(volume, vec3(position - vec3(0, delta, 0))).x;
			sample1.z = texture3D(volume, vec3(position - vec3(0, 0, delta))).x;
			
			if(transferFunctionOn == 1) {
			
				alpha1.x = texture2D(transferFunction, vec2(sample1.x, sample1.x)).a;
				alpha1.y = texture2D(transferFunction, vec2(sample1.y, sample1.y)).a;
				alpha1.z = texture2D(transferFunction, vec2(sample1.z, sample1.z)).a;
			
			}

		}

		//central difference and normalization
		vec3 N;
		vec3 d1, d2;
		
		if(transferFunctionOn == 1) {
		
			if(forwardDifference == 1) {
				d1 = alpha2;
				d2 = vec3(scalar);
			} else {
				d1 = alpha1;
				d2 = alpha2;
			}
		
		} else {
		
			if(forwardDifference == 1) {
				d1 = sample2;
				d2 = vec3(scalar);
			} else {
				d1 = sample1;
				d2 = sample2;
			}
		
		}
					
		N = normalize(d2 - d1);
		vec3 L = normalize(gl_LightSource[0].position.xyz - v); 
		vec3 V = normalize(-v);

		if(useIBL == 1)
			scalar.rgb += diffuseIBL(V) + specularIBL(N, V);
		else
			scalar.rgb += BlinnPhongShading(L, N, V).rgb;

	}

	return scalar;
}

vec4 efficientTriCubicInterpolation(sampler3D texture, vec3 texCoord)
{

	vec3 texelSize;
	ivec3 voxelLength = textureSize(texture, 0);
	texelSize = 1.0 / vec3(voxelLength);

	vec3 coord_grid = texCoord * vec3(voxelLength) - 0.5;
	vec3 index = floor(coord_grid);
	vec3 fraction = coord_grid - index;

	vec3 one_frac = 1.0 - fraction;
	vec3 one_frac2 = one_frac * one_frac;
	vec3 fraction2 = fraction * fraction;

	vec3 w0 = (1.0/6.0) * one_frac2 * one_frac;
	vec3 w1 = (2.0/3.0) - 0.5 * fraction2 * (2.0 - fraction);
	vec3 w2 = (2.0/3.0) - 0.5 * one_frac2 * (2.0 - one_frac);
	vec3 w3 = (1.0/6.0) * fraction2 * fraction;

	vec3 g0 = w0 + w1;
	vec3 g1 = w2 + w3;
	vec3 h0 = texelSize * ((w1 / g0) - 0.5 + index);  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	vec3 h1 = texelSize * ((w3 / g1) + 1.5 + index);

	vec4 tex000 = texture3D(texture, vec3(h0.x, h0.y, h0.z));
	vec4 tex100 = texture3D(texture, vec3(h1.x, h0.y, h0.z));
	tex000 = mix(tex000, tex100, g1.x);  //weight along the x-direction
	vec4 tex010 = texture3D(texture, vec3(h0.x, h1.y, h0.z));
	vec4 tex110 = texture3D(texture, vec3(h1.x, h1.y, h0.z));
	tex010 = mix(tex010, tex110, g1.x);  //weight along the x-direction
	tex000 = mix(tex000, tex010, g1.y);  //weight along the y-direction
	vec4 tex001 = texture3D(texture, vec3(h0.x, h0.y, h1.z));
	vec4 tex101 = texture3D(texture, vec3(h1.x, h0.y, h1.z));
	tex001 = mix(tex001, tex101, g1.x);  //weight along the x-direction
	vec4 tex011 = texture3D(texture, vec3(h0.x, h1.y, h1.z));
	vec4 tex111 = texture3D(texture, vec3(h1.x, h1.y, h1.z));
	tex011 = mix(tex011, tex111, g1.x);  //weight along the x-direction
	tex001 = mix(tex001, tex011, g1.y);  //weight along the y-direction
	return mix(tex000, tex001, g1.z);  //weight along the z-direction	

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

	//Initialize accumulated color and opacity
	vec4 value = vec4(0, 0, 0, 0);
	vec2 scalar = vec2(0, 0);
	vec4 src = vec4(0, 0, 0, 0);
	vec4 dst = vec4(0, 0, 0, 0);
	
	//Determine volume entry position
	vec4 rayStart = texture2D(frontFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	vec4 rayEnd = texture2D(backFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	if(rayStart == rayEnd)
		discard;
	vec3 position = vec3(rayStart);
	vec3 direction = vec3(rayEnd) - vec3(rayStart);
	float len = length(direction); // the length from front to back is calculated and used to terminate the ray
    direction = normalize(direction);
	if(stochasticJithering == 1)
		position = position + direction * texture2D(noise, gl_FragCoord.xy / 256.0).x/16.0;
	float dirLength = length(direction);
	
	//Loop for ray traversal
	float maxStepSize = 0.04; //2.f/50.f
	float accLength = 0.0;
	vec4 maxOpacity;

	//other stuff
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
					if(value.a > 0.075) {
						if(clip) discard;
						else firstHit = true;
					}
				}

			}

		}

		if(maxOpacity.g > 0.0) {

			if(!clip) {

				//Data access to scalar value in 3D volume texture
				if(triCubicInterpolation == 1) {
					value = efficientTriCubicInterpolation(volume, position);
				} else {
					value = texture3D(volume, position);
				}
				
				if(NonPolygonalIsoSurfaceOn == 1) {
					
					vec3 s = vec3(-stepSize * 0.5, -stepSize * 0.5, -stepSize * 0.5);
					position = position + direction * s;
					value = texture3D(volume, position);
					if(value.a > 0.1) s *= 0.5;
					else	s *= -0.5;
					position = position + direction * s;
					value = texture3D(volume, position);
					
				}

				if(transferFunctionOn == 1) {

					scalar.y = value.a;
					//Lookup in pre-integration table
					value = texture2D(transferFunction, scalar.xy);

				}

				if(BlinnPhongShadingOn == 1)
					value = computeIllumination(value, position);

				if(NonPolygonalIsoSurfaceOn == 1) {
					
					if(value.a > isosurfaceThreshold) {
						gl_FragColor = 3 * value;
						return;
					}
					
				} else {
					
					//Front-to-back compositing
					if(MIP == 0) {
						if(value.a > 0.1)
							dst = (1.0 - dst.a) * value + dst;
					} else
						dst = max(dst, value);

				}

				if(transferFunctionOn == 1) {
					//Save previous scalar value
					scalar.x = scalar.y;
				}

			}

			//Advance ray position along ray direction
			if(clippingOcclusion) {
			
				if(firstHit) {
					position = position + direction * stepSize;
					accLength += dirLength * stepSize;
				} else {
					position = position + direction * 0.004;
					accLength += dirLength * 0.004;
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