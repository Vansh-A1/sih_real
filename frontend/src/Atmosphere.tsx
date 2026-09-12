function Cloud({ className }: { className: string }) {
  return (
    <svg
      className={`cloud ${className}`}
      viewBox="0 0 420 180"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M62 156C30 156 14 138 17 116C20 91 43 80 69 85C64 43 96 23 127 28C151-9 207 6 219 37C244 20 281 36 283 66C312 48 350 60 355 91C410 82 425 147 382 157C298 174 157 172 62 156Z"
        fill="currentColor"
      />
      <path
        d="M65 156C152 163 296 164 382 157C395 154 404 146 408 137C316 153 244 144 218 147C148 158 84 142 20 129C25 145 40 156 65 156Z"
        fill="#dceceb"
        fillOpacity=".3"
      />
    </svg>
  );
}

export default function Atmosphere({
  active = false,
  reveal = false,
}: {
  active?: boolean;
  reveal?: boolean;
}) {
  return (
    <div
      className={`atmosphere ${active ? "atmosphere--active" : ""} ${reveal ? "atmosphere--reveal" : ""}`}
      aria-hidden="true"
    >
      <div className="cloud-layer cloud-layer--far">
        <Cloud className="cloud--1" />
        <Cloud className="cloud--2" />
      </div>
      <div className="cloud-layer cloud-layer--near">
        <Cloud className="cloud--3" />
        <Cloud className="cloud--4" />
        <Cloud className="cloud--5" />
        <Cloud className="cloud--6" />
      </div>
      <svg className="wind wind--1" viewBox="0 0 500 140">
        <path d="M5 89C89 88 107 28 184 31S290 124 365 84S425 13 494 23" />
        <path d="M51 105C123 91 122 52 181 54" />
      </svg>
      <svg className="wind wind--2" viewBox="0 0 500 140">
        <path d="M5 89C89 88 107 28 184 31S290 124 365 84S425 13 494 23" />
      </svg>
      <svg className="wind wind--3" viewBox="0 0 500 140">
        <path d="M5 89C89 88 107 28 184 31S290 124 365 84S425 13 494 23" />
      </svg>
    </div>
  );
}
