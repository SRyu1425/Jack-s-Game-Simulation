# Jack-s-Game-Simulation

<img width="1408" height="1076" alt="Image" src="https://github.com/user-attachments/assets/d8aec772-41c1-46fa-b1bb-2ddb4cdb7f0d" />

For this project, I implemented Jack’s Game. The drawing above is a model of all the frames used as well as the configuration variables, q. The bottom left corner is the world reference frame, and each of the box and jack objects have five frames associated with it. There is a single frame at the center of each object, as well as four frames at each of the sides. All rigid body transformations and dynamics calculations are based on these 11 frames. 

From a high level overview, this simulation models a smaller jack bouncing around in a box/cup. Both objects are suspended in gravity, while the box has additional external forces acting on it. Specifically, I chose to have the box both oscillate side to side in the x direction and spin using sinusoidal functions. The final behavior shown in the animation makes sense: when not elastically impacting anything, the jack and box sink due to gravity, and the box shows forced motion as well. When inspecting closely, you can even see that the box feels the impact slightly and bounces in the opposite direction as the jack, which is reasonable when considering impact affecting both bodies.

To solve this problem, I started off by implementing a few helper functions to make the linear algebra streamlined. Because the motion is planar, each SE(3) is really only rotating in z and translating in x,y. We embed it in 4x4 so that inversion and multiplication remain straightforward. In 2D, you could have used a 3x3 SE(2) form, but here we keep a 4x4 matrix and ignore z. Using these functions, I generated the SE(3) rigid body transformations between the frames in my system, shown below. 

<img width="1370" height="899" alt="Image" src="https://github.com/user-attachments/assets/8a12d17a-9a0f-4677-a3bc-219c6d6b317b" />

Using these transformations, I calculated the body velocities for both the jack and box. Because this system is 2D planar motion, we can simplify the inertia tensor to just a single rotational inertia value about the z-axis for the rotational kinetic energy components. By simplifying the representation of both objects to a single point mass, I calculated the rotational inertia to be equal to mr2. Combining these components into the kinetic energy equation, I then calculated the Lagrangian of the system. 

<img width="728" height="89" alt="Image" src="https://github.com/user-attachments/assets/94546291-18e3-4f3c-a25a-5d2143cecd48" />

The external forces on the system were limited to those on the box: a force in the y-axis to slow the gravitational force on the box, as well as a force/torque on the x and theta values of the box. After solving the forced Euler-Lagrange equations, I calculated the impact equations. 

<img width="378" height="103" alt="Image" src="https://github.com/user-attachments/assets/c50c7612-351b-465b-b00f-b6aa34984417" />

Because q is a 6-dimensional vector, I made Fext to be a 6-dimensional vector describing the forces on each of the configuration variables. This gave me 6 equations and 6 unknowns, which after solving for, gave me 6 expressions for the accelerations of each configuration variable and accompanying lambdify functions.

For impact, I first constructed the phi matrix. For each of the 4 walls, there are 4 ends of the jack which can come into contact with each other. This means that the phi matrix has a total of 16 equations/elements. 

For instance, to detect impact between the top of the jack and the top of the box, I first found gb1j1. This transformation matrix can be thought of as the position of j1 (top of the jack) in the frame of b1 (top of the box). From there, I extracted the y-element of the translation (1,3) as my constraint for that combination. Impact would be defined as when the y-value of gb1j1 > 0. I then repeated this for each end of the jack, and for each wall (keeping in mind that the left and right walls extract the x-value). A key thing to note is that for the bottom and left walls of the box, I used the negative value of x and y such that impact could be simplified to the case of when any of the jack ends become a positive value. 

Then, using the impact update laws below, I built a 16-element list of impact equations and loop over them in code: for each constraint I check whether φ > 0; if so, I solve the seven post-impact unknowns (x_b_dot_plus, y_b_dot_plus, theta_b_dot_plus, x_j_dot_plus, y_j_dot_plus, theta_j_dot_plus, lambda), keep the solution with a non-zero lambda, and update the state vector.

<img width="1004" height="399" alt="Image" src="https://github.com/user-attachments/assets/8eaf329e-0e44-4f05-998a-47948759f5af" />
