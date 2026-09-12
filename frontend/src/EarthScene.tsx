import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { ArrowUpRight, Satellite } from "lucide-react";

type Props = {
  diving: boolean;
  paused: boolean;
  reducedMotion: boolean;
  onEnter: (satellite?: string) => void;
};

// Deliberately simplified continent outlines, in longitude / latitude.
const continents = [
  [
    [-17, 36],
    [-6, 36],
    [11, 37],
    [24, 32],
    [34, 31],
    [35, 23],
    [43, 12],
    [51, 12],
    [43, 0],
    [40, -12],
    [33, -27],
    [19, -35],
    [12, -18],
    [9, -3],
    [-2, 5],
    [-15, 12],
    [-17, 23],
  ],
  [
    [-10, 36],
    [-10, 44],
    [-1, 44],
    [-5, 49],
    [5, 54],
    [8, 59],
    [5, 62],
    [18, 71],
    [30, 70],
    [31, 60],
    [43, 66],
    [61, 69],
    [83, 73],
    [110, 76],
    [141, 70],
    [174, 63],
    [178, 51],
    [151, 49],
    [141, 35],
    [123, 24],
    [121, 6],
    [111, 0],
    [105, 10],
    [100, 17],
    [99, 6],
    [93, 10],
    [88, 23],
    [79, 8],
    [72, 20],
    [63, 25],
    [55, 24],
    [49, 13],
    [43, 13],
    [36, 30],
    [28, 41],
    [24, 36],
    [18, 40],
    [14, 45],
    [10, 44],
    [4, 43],
  ],
  [
    [-168, 71],
    [-144, 70],
    [-130, 58],
    [-124, 49],
    [-123, 39],
    [-113, 28],
    [-104, 22],
    [-97, 16],
    [-86, 10],
    [-82, 8],
    [-87, 20],
    [-81, 25],
    [-81, 31],
    [-72, 43],
    [-55, 51],
    [-62, 60],
    [-78, 63],
    [-92, 73],
    [-118, 76],
    [-150, 72],
  ],
  [
    [-81, 12],
    [-70, 11],
    [-61, 7],
    [-50, -1],
    [-35, -6],
    [-39, -18],
    [-48, -29],
    [-56, -35],
    [-66, -55],
    [-73, -50],
    [-76, -32],
    [-81, -5],
  ],
  [
    [112, -22],
    [116, -34],
    [133, -35],
    [139, -38],
    [153, -28],
    [145, -14],
    [137, -12],
    [130, -15],
    [122, -14],
  ],
  [
    [-52, 59],
    [-43, 60],
    [-21, 75],
    [-27, 82],
    [-47, 84],
    [-63, 76],
  ],
  [
    [46, -13],
    [50, -16],
    [47, -26],
    [43, -24],
  ],
  [
    [-10, 50],
    [-6, 58],
    [0, 59],
    [2, 52],
  ],
  [
    [130, 32],
    [140, 42],
    [146, 45],
    [142, 35],
    [136, 31],
  ],
  [
    [167, -34],
    [176, -39],
    [172, -45],
    [166, -47],
  ],
];

function inPolygon(x: number, y: number, polygon: number[][]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

function makeEarth() {
  const geometry = new THREE.IcosahedronGeometry(2, 14);
  const position = geometry.getAttribute("position");
  const colors: number[] = [];
  const center = new THREE.Vector3();
  const color = new THREE.Color();
  for (let i = 0; i < position.count; i += 3) {
    center.set(0, 0, 0);
    for (let v = 0; v < 3; v++)
      center.add(new THREE.Vector3().fromBufferAttribute(position, i + v));
    center.normalize();
    const lon = THREE.MathUtils.radToDeg(Math.atan2(center.x, center.z));
    const lat = THREE.MathUtils.radToDeg(Math.asin(center.y));
    const land =
      continents.some((polygon) => inPolygon(lon, lat, polygon)) || lat < -77;
    const variation = Math.sin(i * 127.1 + 311.7) * 0.035;
    if (land) color.setHSL(0.26 + variation * 0.3, 0.22, 0.64 + variation);
    else color.setHSL(0.52 + variation * 0.12, 0.29, 0.39 + variation);
    color.convertSRGBToLinear();
    if (Math.abs(lat) > 76) color.set("#d3e4de");
    for (let v = 0; v < 3; v++) {
      colors.push(color.r, color.g, color.b);
    }
  }
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 0.88,
      metalness: 0.02,
    }),
  );
}

function makeSatellite() {
  const group = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({
    color: "#edece2",
    roughness: 0.45,
    metalness: 0.15,
  });
  const gold = new THREE.MeshStandardMaterial({
    color: "#d4bc87",
    roughness: 0.5,
    metalness: 0.3,
  });
  const blue = new THREE.MeshStandardMaterial({
    color: "#355d79",
    metalness: 0.35,
    roughness: 0.45,
  });
  const line = new THREE.MeshStandardMaterial({
    color: "#7b9da7",
    metalness: 0.25,
    roughness: 0.5,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.34, 0.3), white);
  const base = new THREE.Mesh(new THREE.BoxGeometry(0.29, 0.11, 0.31), gold);
  base.position.y = -0.14;
  group.add(body, base);
  for (const side of [-1, 1]) {
    const connector = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.035, 0.035),
      gold,
    );
    connector.position.x = side * 0.24;
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.025, 0.39),
      blue,
    );
    panel.position.x = side * 0.58;
    panel.rotation.x = 0.28;
    group.add(connector, panel);
    for (let i = 0; i < 4; i++) {
      const grid = new THREE.Mesh(
        new THREE.BoxGeometry(0.009, 0.009, 0.38),
        line,
      );
      grid.position.set(side * 0.58 + (i - 1.5) * 0.13, 0.019, 0);
      grid.rotation.x = 0.28;
      group.add(grid);
    }
    const grid = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.012, 0.007),
      line,
    );
    grid.position.set(side * 0.58, 0.021, 0);
    group.add(grid);
  }
  const sensor = new THREE.Mesh(
    new THREE.CylinderGeometry(0.074, 0.09, 0.1, 10),
    blue,
  );
  sensor.rotation.x = Math.PI / 2;
  sensor.position.z = 0.19;
  const dish = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 10, 5, 0, Math.PI * 2, 0, Math.PI / 2),
    white,
  );
  dish.rotation.z = -0.4;
  dish.position.set(0.04, 0.24, 0);
  group.add(sensor, dish);
  return group;
}

function makeCloud() {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: "#e9f1e9",
    flatShading: true,
    roughness: 1,
  });
  [
    [-0.19, 0, 0.16],
    [0, 0.055, 0.21],
    [0.2, 0, 0.14],
  ].forEach(([x, y, size]) => {
    const puff = new THREE.Mesh(
      new THREE.IcosahedronGeometry(size, 1),
      material,
    );
    puff.position.set(x, y, 0);
    puff.scale.z = 0.4;
    group.add(puff);
  });
  return group;
}

export default function EarthScene({
  diving,
  paused,
  reducedMotion,
  onEnter,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const labels = useRef<(HTMLButtonElement | null)[]>([]);
  const flags = useRef({ diving, paused, reducedMotion });
  const hovered = useRef(-1);
  const [fallback, setFallback] = useState(false);
  flags.current = { diving, paused, reducedMotion };

  useEffect(() => {
    const host = container.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "low-power",
      });
    } catch {
      setFallback(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.prepend(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
    camera.position.set(0, 0.5, 8.6);
    camera.lookAt(0, 0, 0);
    scene.add(new THREE.AmbientLight("#d6e9e8", 1.2));
    const sun = new THREE.DirectionalLight("#fff2d9", 2.5);
    sun.position.set(-3, 5, 5);
    scene.add(sun);
    const rim = new THREE.DirectionalLight("#a3cad8", 1.3);
    rim.position.set(4, 1, -3);
    scene.add(rim);
    const world = new THREE.Group();
    world.rotation.z = -0.12;
    scene.add(world);
    const planet = new THREE.Group();
    planet.add(makeEarth());
    const atmosphere = new THREE.Mesh(
      new THREE.IcosahedronGeometry(2.045, 8),
      new THREE.MeshBasicMaterial({
        color: "#bad8dd",
        transparent: true,
        opacity: 0.11,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    );
    planet.add(atmosphere);
    [
      [-1.3, 0.8, 1.65],
      [0.7, 1.45, 1.5],
      [1.6, -0.55, 1.34],
      [-0.55, -1.52, 1.5],
    ].forEach(([x, y, z], i) => {
      const cloud = makeCloud();
      cloud.position.set(x, y, z).normalize().multiplyScalar(2.075);
      cloud.lookAt(cloud.position.clone().multiplyScalar(2));
      cloud.rotation.z = i * 0.4;
      cloud.scale.setScalar(i === 1 ? 1 : 0.8);
      planet.add(cloud);
    });
    world.add(planet);
    const satellites: THREE.Group[] = [];
    const orbitMaterials: THREE.LineBasicMaterial[] = [];
    for (let i = 0; i < 2; i++) {
      const orbit = new THREE.Group();
      orbit.rotation.set(
        i === 0 ? 0.62 : 1.14,
        i === 0 ? 0.17 : -0.5,
        i === 0 ? -0.39 : 0.61,
      );
      const points = Array.from({ length: 201 }, (_, step) => {
        const angle = (step / 200) * Math.PI * 2;
        return new THREE.Vector3(
          2.9 * Math.cos(angle),
          0,
          2.9 * Math.sin(angle),
        );
      });
      const material = new THREE.LineBasicMaterial({
        color: "#b9d5d6",
        transparent: true,
        opacity: i === 0 ? 0.26 : 0.13,
      });
      const ring = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        material,
      );
      const satellite = makeSatellite();
      satellite.scale.setScalar(i === 0 ? 0.72 : 0.55);
      orbit.add(ring, satellite);
      world.add(orbit);
      satellites.push(satellite);
      orbitMaterials.push(material);
    }
    let width = 1;
    let height = 1;
    const resize = new ResizeObserver(() => {
      width = host.clientWidth;
      height = host.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    });
    resize.observe(host);
    let raf = 0;
    let elapsed = 0;
    let last = performance.now();
    let diveProgress = 0;
    const projected = new THREE.Vector3();
    const render = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const {
        diving: isDiving,
        paused: isPaused,
        reducedMotion: reduce,
      } = flags.current;
      if (!isPaused && !reduce && !document.hidden) elapsed += dt;
      planet.rotation.y = -0.28 + elapsed * 0.035;
      world.position.y =
        reduce || isPaused ? 0 : Math.sin(elapsed * 0.4) * 0.035;
      diveProgress = THREE.MathUtils.damp(
        diveProgress,
        isDiving ? 1 : 0,
        reduce ? 40 : 2.3,
        dt,
      );
      const orbitDistance = Math.max(
        8.6,
        3.45 / (Math.tan(THREE.MathUtils.degToRad(37 / 2)) * camera.aspect),
      );
      camera.position.z = orbitDistance - diveProgress * (orbitDistance - 2);
      camera.position.y = 0.5 + diveProgress * 0.15;
      camera.lookAt(0, 0.05, 0);
      satellites.forEach((satellite, i) => {
        const angle =
          (i === 0 ? 0.42 : 3.75) + elapsed * (i === 0 ? 0.11 : -0.075);
        satellite.position.set(2.9 * Math.cos(angle), 0, 2.9 * Math.sin(angle));
        satellite.rotation.set(0.25 + elapsed * 0.045, -angle + 0.3, 0.25);
        const targetScale =
          (i === 0 ? 0.72 : 0.55) * (hovered.current === i ? 1.18 : 1);
        satellite.scale.lerp(
          new THREE.Vector3(targetScale, targetScale, targetScale),
          0.09,
        );
        orbitMaterials[i].opacity =
          hovered.current === i ? 0.48 : i === 0 ? 0.26 : 0.13;
        const label = labels.current[i];
        if (label) {
          satellite.getWorldPosition(projected);
          // Hide labels while the planet occludes a satellite.
          const visible =
            projected.z > 0.1 || Math.hypot(projected.x, projected.y) > 2.17;
          projected.project(camera);
          label.style.transform = `translate(${(projected.x * 0.5 + 0.5) * width}px, ${(-projected.y * 0.5 + 0.5) * height}px)`;
          label.style.setProperty(
            "--tooltip-left",
            (projected.x * 0.5 + 0.5) * width > width - 155 ? "-78px" : "24px",
          );
          label.style.opacity = visible && !isDiving ? "1" : "0";
          label.style.pointerEvents = visible && !isDiving ? "auto" : "none";
          label.tabIndex = visible && !isDiving ? 0 : -1;
        }
      });
      if (!document.hidden) renderer.render(scene, camera);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(raf);
      resize.disconnect();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div
      ref={container}
      className={`earth-scene ${diving ? "earth-scene--diving" : ""}`}
    >
      {fallback ? (
        <button
          className="fallback-earth"
          onClick={() => onEnter("01")}
          aria-label="Enter the RezX workspace"
        >
          <span />
          <Satellite size={28} />
        </button>
      ) : (
        [0, 1].map((i) => (
          <button
            key={i}
            ref={(el) => {
              labels.current[i] = el;
            }}
            className={`satellite-hotspot satellite-hotspot--${i + 1}`}
            onMouseEnter={() => {
              hovered.current = i;
            }}
            onMouseLeave={() => {
              hovered.current = -1;
            }}
            onFocus={() => {
              hovered.current = i;
            }}
            onBlur={() => {
              hovered.current = -1;
            }}
            onClick={() => onEnter(`0${i + 1}`)}
            aria-label={`Select satellite 0${i + 1} and enter the workspace`}
          >
            <span className="satellite-target" />
            <span className="satellite-tooltip">
              <span>
                <i /> Satellite 0{i + 1}
                <ArrowUpRight size={13} />
              </span>
              <small>Earth observation</small>
            </span>
          </button>
        ))
      )}
    </div>
  );
}
