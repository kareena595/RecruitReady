// app/page.tsx
import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="min-h-screen min-w-screen flex flex-col items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-blue-100 px-6 py-20 text-center">
      {/* Hero Section */}
      <div className="max-w-3xl">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          Ace Your Interviews with{" "}
          <span className="text-indigo-600">AI-Powered Practice</span>
        </h1>

        <p className="text-lg md:text-xl text-gray-700 mb-10">
          Practice with an intelligent virtual interviewer that analyzes your
          answers, tone, and body language — helping you perform your best when
          it matters most.
        </p>

        <Link
          href="/app"
          className="inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-lg px-8 py-4 rounded-full shadow-lg transition"
        >
          Launch
        </Link>
      </div>

      {/* Features Section */}
      <div className="mt-24 grid gap-8 md:grid-cols-3 max-w-5xl">
        <FeatureCard
          title="Realistic AI Interviews"
          description="Simulate real interview scenarios powered by natural conversation AI."
        />
        <FeatureCard
          title="Body Language Analysis"
          description="Our computer vision model observes your posture, gestures, and confidence."
        />
        <FeatureCard
          title="Instant Personalized Feedback"
          description="Get clear, actionable insights to improve how you answer and present."
        />
      </div>
    </main>
  );
}

interface FeatureCardProps {
  title: string;
  description: string;
}

function FeatureCard({ title, description }: FeatureCardProps) {
  return (
    <div className="bg-white p-8 rounded-2xl shadow-md hover:shadow-lg transition">
      <h3 className="text-xl font-semibold text-gray-800 mb-3">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}
